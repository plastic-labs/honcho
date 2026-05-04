# Honcho CrewAI Integration

Build CrewAI agents with persistent memory and reasoning capabilities powered by Honcho.

## Installation

```bash
uv add honcho-crewai crewai python-dotenv
```

CrewAI currently supports Python `>=3.10,<3.14`; this package follows the same range.

## Quick Start

```python
from crewai import Agent, Crew, Memory, Process, Task
from honcho import Honcho
from honcho_crewai import HonchoMemoryStorage

honcho = Honcho(workspace_id="crewai-demo")
storage = HonchoMemoryStorage(
    peer_id="user-123",
    session_id="session-123",
    honcho_client=honcho,
)
memory = Memory(storage=storage)

memory.remember(
    "The user is learning Python and wants to build web applications.",
    scope="/users/user-123",
    categories=["preferences"],
    metadata={"source": "onboarding"},
)

agent = Agent(
    role="Programming Mentor",
    goal="Help users learn programming by remembering their interests and progress",
    backstory="You are a patient programming mentor.",
)

task = Task(
    description="Suggest a Python web project that matches the user's interests.",
    expected_output="A specific project suggestion with a brief explanation",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    memory=memory,
)

result = crew.kickoff()
print(result.raw)
```

## Features

- `HonchoMemoryStorage`: CrewAI unified `Memory` storage backend.
- `HonchoStorage`: compatibility adapter for older CrewAI `ExternalMemory` usage.
- `HonchoGetContextTool`, `HonchoDialecticTool`, and `HonchoSearchTool` for explicit Honcho memory retrieval.
- Lazy Honcho peer/session handles, matching the latest Honcho SDK get-or-create behavior.

## Documentation

For guides and API reference, visit [docs.honcho.dev](https://docs.honcho.dev/v3/guides/integrations/crewai).

## License

AGPL-3.0-or-later
