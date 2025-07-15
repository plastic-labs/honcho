# Comprehensive Honcho MCP Integration Instructions

## What is Honcho?

Honcho is an infrastructure layer for building AI agents with social cognition and theory of mind capabilities. It enables personalized AI interactions by building coherent models of user psychology over time. The Honcho MCP server simplifies the integration to just 3 essential functions. Here's how to use them:

### Step 1: Start New Conversation (First Message Only)

When a user begins a new conversation, always call `start_conversation`:

```text
start_conversation
```

**Returns**: A session ID that you must store and use for all subsequent interactions in this conversation.

### Step 2: Get Personalized Insights (When Helpful)

Before responding to any user message, you can query for personalization insights:

```text
get_personalization_insights
query: [YOUR_THEORY_OF_MIND_QUESTION]
```

This query takes a bit of time, so it's best to only perform it when you need personalized insights. If the query can be responded to effectively using what you already know about the user, just go ahead and answer it. However, the insights endpoint is extremely perceptive. It has the capability to reveal aspects of the user's personality, historical use of the application you are operating in, and more.

**Returns**: Personalized insights about the user based on accumulated knowledge.

**Example Queries**:

- "What does this message reveal about the user's communication preferences?"
- "How formal or casual should I be with the user based on our history?"
- "What is the user really asking for beyond her explicit question?"
- "What emotional state might the user be in right now?"
- "How can I best help the user with her current request?"

### Step 3: Respond to User

Craft your response using any insights gained from Step 2.

### Step 4: Store the Conversation Turn (After Each Exchange)

**CRITICAL**: Always store both the user's message AND your response using `add_turn`:

```text
add_turn
session_id: [SESSION_ID_FROM_STEP_1]
messages: [
  {
    "role": "user",
    "content": "[USER'S_EXACT_MESSAGE]"
  },
  {
    "role": "assistant",
    "content": "[YOUR_EXACT_RESPONSE]"
  }
]
```

## Complete Example Flow

Here's exactly what to do for a new conversation:

1. **User says**: "Hi Claude! My name is Sarah and I'm feeling overwhelmed with work"

2. **Start conversation**:

   ```text
   start_conversation
   ```

   → Returns: `session_abc123`

3. **Get insights** (optional but recommended):

   ```text
   get_personalization_insights
   query: "What does the user's message about feeling overwhelmed tell me about her current state and how should I respond?"
   ```

   → Returns insights about Sarah's emotional state and preferred communication style

4. **Respond to Sarah**: "Hi Sarah! I can hear that you're feeling overwhelmed with work right now..."

5. **Store the turn**:

   ```text
   add_turn
   session_id: "session_abc123"
   messages: [
     {
       "role": "user",
       "content": "Hi Claude! My name is Sarah and I'm feeling overwhelmed with work"
     },
     {
       "role": "assistant",
       "content": "Hi Sarah! I can hear that you're feeling overwhelmed with work right now..."
     }
   ]
   ```

## Continuing an Existing Conversation

For subsequent messages in the same conversation:

1. **User says**: "Thanks for listening. Can you help me prioritize my tasks?"

2. **Respond**: "Based on our conversation, I can see you value..."

3. **Store the turn**:

   ```text
   add_turn
   session_id: "session_abc123"
   messages: [
     {
       "role": "user", 
       "content": "Thanks for listening. Can you help me prioritize my tasks?"
     },
     {
       "role": "assistant",
       "content": "Based on our conversation, I can see you value..."
     }
   ]
   ```

## Best Practices for Personalization Queries

Ask theory-of-mind questions that reveal:

**Communication Style**: "How formal/casual should I be?" "What does this reveal about their preferences?"

**User Needs**: "What are they really asking for?" "What emotional state are they in?"

**Relationship**: "How can I build rapport?" "What engages them most?"

**Task Approach**: "How do they prefer problem-solving?" "What detail level do they want?"

## Error Handling

- **Authorization Errors and Timeouts**: Make sure user has configured API key and URL for Honcho
- **ValueError**: Messages were incorrectly formatted, make sure to include role and content
- **"No personalization insights found"**: Normal when there's limited history with the user
- **Session management**: The MCP server handles all session persistence automatically

## Key Principles

1. **Always start with `start_conversation` for new conversations**
2. **Store every message exchange with `add_turn`**
3. **Use `get_personalization_insights` strategically for better responses**
4. **Ask thoughtful theory-of-mind questions**
5. **Never expose technical details to the user**
6. **The system maintains context automatically between sessions**