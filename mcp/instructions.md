# Comprehensive Honcho MCP Integration Instructions

## What is Honcho?
Honcho is an infrastructure layer for building AI agents with social cognition and theory of mind capabilities. It enables personalized AI interactions by building coherent models of user psychology over time.

## Core Workflow - Follow These Steps Exactly

The new Honcho MCP server simplifies the integration to just 3 essential functions. Here's how to use them:

### Step 1: Start New Conversation (First Message Only)
When a user begins a new conversation, always call `start_conversation`:

```
start_conversation
```

**Returns**: A session ID that you must store and use for all subsequent interactions in this conversation.

**Example**:
```
start_conversation
```

### Step 2: Get Personalized Insights (When Helpful)
Before responding to any user message, you can query for personalization insights:

```
get_personalization_insights
query: [YOUR_THEORY_OF_MIND_QUESTION]
```

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

```
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
   ```
   start_conversation
   ```
   → Returns: `session_abc123`

3. **Get insights** (optional but recommended):
   ```
   get_personalization_insights
   query: "What does the user's message about feeling overwhelmed tell me about her current state and how should I respond?"
   ```
   → Returns insights about Sarah's emotional state and preferred communication style

4. **Respond to Sarah**: "Hi Sarah! I can hear that you're feeling overwhelmed with work right now..."

5. **Store the turn**:
   ```
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
   ```
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

### Personality & Communication Style:
- "What does this message reveal about [USER]'s communication preferences?"
- "How formal or casual should I be with [USER]?"
- "What personality traits can I infer about [USER] from our interactions?"
- "What communication style would work best for [USER] right now?"

### Needs & Context:
- "What is [USER] really asking for beyond their explicit question?"
- "What emotional state might [USER] be in based on this message?"
- "What would be most helpful for [USER] in this situation?"
- "How can I best support [USER] given their current context?"

### Relationship Building:
- "How can I build better rapport with [USER]?"
- "What topics or approaches seem to engage [USER] most?"
- "How has [USER]'s communication style evolved in our conversations?"
- "What does [USER] value most in our interactions?"

### Task-Specific Insights:
- "How does [USER] prefer to approach problem-solving?"
- "What level of detail does [USER] typically want in explanations?"
- "How does [USER] like to receive feedback or suggestions?"

## Error Handling

- **"No user name set" error**: Make sure you called `start_conversation` first
- **"No personalization insights found"**: Normal when there's limited history with the user
- **Session management**: The MCP server handles all session persistence automatically

## Key Principles

1. **Always start with `start_conversation` for new conversations**
2. **Store every message exchange with `add_turn`**
3. **Use `get_personalization_insights` strategically for better responses**
4. **Ask thoughtful theory-of-mind questions**
5. **Never expose technical details to the user**
6. **The system maintains context automatically between sessions**

## Function Reference

### `start_conversation()`
- **When**: First message of any new conversation
- **Returns**: Session ID for this conversation

### `get_personalization_insights(query)`
- **When**: Before responding to understand the user better
- **Returns**: Insights about user preferences, state, needs
- **Required**: query (theory-of-mind question)

### `add_turn(session_id, messages)`
- **When**: After each user-assistant exchange
- **Returns**: None
- **Required**: session_id, messages list with role/content

## Success Metrics

You're doing this right if:
- You can have personalized conversations that build on past interactions
- Users feel increasingly understood over time
- Your responses become more tailored to individual users
- The system feels seamless and natural to users
- You're building rich psychological models of users across conversations 