# Honcho Python SDK

The official Python library for the [Honcho](https://github.com/plastic-labs/honcho) conversational memory platform. Honcho provides tools for managing peers, sessions, and conversation context across multi-party interactions, enabling advanced conversational AI applications with persistent memory and theory-of-mind capabilities.

## Installation

```bash
pip install honcho-ai
```

## Quick Start

```python
from honcho import Honcho

# Initialize client
client = Honcho(api_key="your-api-key")

# Create peers (participants in conversations)
alice = client.peer("alice")
bob = client.peer("bob")

# Create a session for group conversations
session = client.session("conversation-1")

# Add messages to the session
session.add_messages([
    alice.message("Hello, Bob!"),
    bob.message("Hi Alice, how are you?")
])

# Wait for deriver to process all messages (only necessary if very recent messages are critical to query)
client.poll_deriver_status()

# Query conversation context
response = alice.chat("What did Bob say to the user?")
print(response)
```

## Core Concepts

### Peers

Peers represent participants in conversations.

```python
# Create peers
assistant = client.peer("assistant")
user = client.peer("user-123")

# Chat with global context
response = user.chat("What did I talk about yesterday?")

# Chat with perspective of another peer
response = user.chat("Does the assistant know my preferences?", target=assistant)
```

### Sessions

Sessions group related conversations and messages:

```python
# Create a session
session = client.session("project-discussion")

# Add peers to session
session.add_peers([alice, bob])

# Add messages
session.add_messages([
    alice.message("Let's discuss the project timeline"),
    bob.message("I think we need two more weeks")
])

# Get conversation context
context = session.get_context()
```

### Messages and Context

Retrieve and use conversation history:

```python
# Get messages from a session
messages = session.get_messages()

# Convert to OpenAI format for further prompting
openai_messages = context.to_openai(assistant="assistant")

# Convert to Anthropic format for further prompting
anthropic_messages = context.to_anthropic(assistant="assistant")
```

### Async Support

```python
from honcho import AsyncHoncho

async def main():
    client = AsyncHoncho(api_key="your-api-key")
```

### Metadata Management

```python
# Set peer metadata
user.set_metadata({"location": "San Francisco", "preferences": {"theme": "dark"}})

# Session metadata
session.set_metadata({"topic": "project-planning", "priority": "high"})
```

### Multi-Perspective Queries

```python
# Alice's view of what Bob knows
response = alice.chat("Does Bob remember our discussion about the budget?", target=bob)

# Session-specific perspective
response = alice.chat("What does Bob think about this project?",
                     target=bob,
                     session_id=session.id)
```

## Reasoning Artifacts (Read-Only)

Honcho generates reasoning artifacts during periodic "dreams" where reasoning agents analyze observed patterns and form hypotheses, predictions, and inductions. These artifacts provide transparency into the system's understanding of peers.

### Hypotheses

Hypotheses are explanatory theories about observed patterns:

```python
# Get active hypotheses for a peer (via client)
hypotheses = client.get_hypotheses(
    observer="user_123",
    observed="user_123",
    status="active"
)

# Get hypotheses using Peer convenience method
peer = client.peer("user_123")
hypotheses = peer.get_hypotheses(status="active")

# Get hypotheses about another peer
hypotheses = peer.get_hypotheses(target="user_456")

# Get hypothesis details
hypothesis = client.get_hypothesis("hyp_abc123")
print(hypothesis.content)
print(hypothesis.confidence)

# Get hypothesis evolution tree
genealogy = client.get_hypothesis_genealogy("hyp_abc123")
print(f"Parents: {len(genealogy['parents'])}")
print(f"Children: {len(genealogy['children'])}")
```

### Predictions

Predictions are testable claims derived from hypotheses:

```python
# List predictions
predictions = client.get_predictions(
    hypothesis_id="hyp_abc123",
    status="unfalsified"
)

# Semantic search for similar predictions
similar = client.search_predictions(
    "prefers dark mode over light mode",
    hypothesis_id="hyp_abc123"
)

# Get falsification traces for a prediction
traces = client.get_prediction_traces("pred_xyz789")
```

### Falsification Traces

Traces record the falsification process:

```python
# List traces
traces = client.get_traces(
    prediction_id="pred_xyz789",
    final_status="unfalsified"
)

# Get trace details
trace = client.get_trace("trace_123")
print(f"Search queries: {trace.search_queries}")
print(f"Contradicting premises: {trace.contradicting_premise_ids}")
print(f"Reasoning chain: {trace.reasoning_chain}")
```

### Inductions

Inductions are stable patterns extracted from unfalsified predictions:

```python
# Get high-confidence patterns (via client)
inductions = client.get_inductions(
    observer="user_123",
    observed="user_123",
    confidence="high"
)

# Get patterns using Peer convenience method
peer = client.peer("user_123")
patterns = peer.get_inductions(confidence="high")

# Get patterns about another peer
patterns = peer.get_inductions(
    target="user_456",
    pattern_type="behavioral"
)

# Get pattern sources (provenance)
sources = client.get_induction_sources("ind_abc123")
print(f"Based on {len(sources['source_predictions'])} predictions")
print(f"From {len(sources['source_premises'])} observations")
```

**Note**: All reasoning artifacts are read-only. They are generated exclusively during reasoning dreams and cannot be created or modified via the API.

### Type Safety

The SDK includes TypedDict definitions for all reasoning artifacts, providing IDE autocomplete and type checking:

```python
from honcho import Honcho
from honcho import Hypothesis, Induction, Prediction, FalsificationTrace

client = Honcho()
peer = client.peer("user-123")

# Type hints provide IDE autocomplete
hypotheses: list[Hypothesis] = peer.get_hypotheses(status="active")
for hyp in hypotheses:
    print(hyp["content"])  # IDE knows this key exists
    print(hyp["confidence_score"])  # Autocomplete available
    print(hyp["tier"])  # Type-safe access

# Works with all reasoning artifact types
inductions: list[Induction] = peer.get_inductions(confidence="high")
for pattern in inductions:
    print(f"Pattern: {pattern['content']}")
    print(f"Type: {pattern['pattern_type']}")
    print(f"Stability: {pattern['stability_score']}")
```

Available types: `Hypothesis`, `Prediction`, `FalsificationTrace`, `Induction`, `HypothesisGenealogy`, `InductionSources`

## Configuration

### Environment Variables

```bash
export HONCHO_API_KEY="your-api-key"
export HONCHO_BASE_URL="https://api.honcho.dev"  # Optional
export HONCHO_WORKSPACE_ID="your-workspace"  # Optional
```

### Client Options

```python
client = Honcho(
    api_key="your-api-key",
    environment="production",  # or "local", "demo"
    workspace_id="custom-workspace",
    base_url="https://api.honcho.dev"
)
```

## Examples

Check out the `examples/` directory for complete usage examples:

- `example.py` - Comprehensive feature demonstration
- `chat.py` - Basic multi-peer chat
- `async_example.py` - Async/await usage
- `search.py` - Context search and retrieval
- `reasoning_artifacts.py` - Querying hypotheses, predictions, traces, and inductions

## License

Apache 2.0 - see [LICENSE](../../LICENSE) for details.

## Support

- [Documentation](https://docs.honcho.dev)
- [GitHub Issues](https://github.com/plastic-labs/honcho-sdks/issues)
- [Discord Community](https://discord.gg/honcho)
