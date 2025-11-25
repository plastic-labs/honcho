# Honcho CrewAI Integration

Build CrewAI agents with persistent memory and theory-of-mind capabilities powered by Honcho.

## Installation

```bash
pip install honcho-crewai
```

## Quick Start

```python
from crewai import Agent, Task, Crew, Process
from crewai.memory.external.external_memory import ExternalMemory
from honcho_crewai import HonchoStorage

# Initialize Honcho storage
storage = HonchoStorage(user_id="user-123")
external_memory = ExternalMemory(storage=storage)

# Create agent with memory
agent = Agent(
    role="AI Assistant",
    goal="Help users with persistent memory",
    backstory="You remember past conversations.",
)

# Create crew with external memory
crew = Crew(
    agents=[agent],
    tasks=[task],
    external_memory=external_memory
)
```

## Features

- **Automatic Memory**: CrewAI agents automatically store and retrieve conversation context
- **Semantic Search**: Find relevant past messages using vector similarity
- **Theory of Mind**: Query what the system knows about users via the Dialectic API
- **Multi-Agent Support**: Give each agent distinct memory and identity
- **Tools Integration**: `HonchoGetContextTool`, `HonchoDialecticTool`, and `HonchoSearchTool` for explicit memory control

## Documentation

For comprehensive guides, examples, and API reference, visit:
**[https://docs.honcho.dev/v2/integrations/crewai](https://docs.honcho.dev/v2/integrations/crewai)**

## Examples

Check out complete examples in the [GitHub repository](https://github.com/plastic-labs/honcho/tree/main/examples/crewai/python/examples).

## License

AGPL-3.0-or-later

## Support

- Report issues: [GitHub Issues](https://github.com/plastic-labs/honcho/issues)
- Documentation: [docs.honcho.dev](https://docs.honcho.dev)
- Website: [honcho.dev](https://honcho.dev)
