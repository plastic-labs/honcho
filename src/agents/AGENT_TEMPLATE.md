# Agent Directory Template

This document defines the standard directory structure for all Honcho agents. Following this template ensures consistency across the codebase and makes agents easier to understand, test, and maintain.

## Standard Directory Structure

```
agent_name/
├── __init__.py          # Package initialization and exports
├── agent.py             # Main agent class (inherits from BaseAgent)
├── config.py            # Agent-specific configuration (optional)
├── prompts.py           # Agent-specific prompt templates (optional)
├── tools.py             # Agent-specific tool definitions (optional)
└── README.md            # Agent documentation (optional)
```

## File Descriptions

### `__init__.py` (Required)

Package initialization that exports the main agent class and any public APIs.

**Template:**
```python
"""
[Agent Name] - [Brief description of agent's purpose]

This module implements [describe what the agent does].
"""

from .agent import AgentNameAgent

__all__ = [
    "AgentNameAgent",
]
```

### `agent.py` (Required)

Main agent implementation that inherits from `BaseAgent`.

**Template:**
```python
"""
Main implementation of the [Agent Name] agent.
"""

import logging
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.shared import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class AgentNameAgent(BaseAgent):
    """
    [Brief description of agent's purpose and functionality]

    This agent is responsible for [describe responsibilities].

    Attributes:
        db: Database session for agent operations
        config: Agent configuration
        [additional agent-specific attributes]
    """

    def __init__(
        self,
        db: AsyncSession,
        config: AgentConfig | None = None,
        **kwargs
    ):
        """
        Initialize the [Agent Name] agent.

        Args:
            db: SQLAlchemy async database session
            config: Agent-specific configuration
            **kwargs: Additional agent-specific parameters
        """
        super().__init__(db, config, **kwargs)
        # Initialize agent-specific attributes here

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.

        Args:
            input_data: Dictionary containing:
                - [list required input fields]

        Returns:
            Dictionary containing:
                - [list output fields]

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If agent execution fails
        """
        # Implement agent logic here
        logger.info(f"[{self.agent_type}] Executing with input: {input_data.keys()}")

        # Example implementation:
        result = await self._process_input(input_data)

        return {
            "success": True,
            "result": result,
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate the input data before execution.

        Args:
            input_data: Dictionary containing input data to validate

        Returns:
            True if input is valid

        Raises:
            ValueError: If input validation fails with a descriptive error message
        """
        # Implement validation logic
        required_fields = []  # Define required fields

        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        return True

    async def _process_input(self, input_data: Dict[str, Any]) -> Any:
        """
        Private helper method for processing input.

        Args:
            input_data: Validated input data

        Returns:
            Processed result
        """
        # Implement processing logic
        pass
```

### `config.py` (Optional)

Agent-specific configuration class that extends `AgentConfig`.

**Template:**
```python
"""
Configuration for the [Agent Name] agent.
"""

from pydantic import Field

from src.agents.shared import AgentConfig


class AgentNameConfig(AgentConfig):
    """
    Configuration for the [Agent Name] agent.

    Extends the base AgentConfig with agent-specific parameters.
    """

    # Agent-specific configuration fields
    param_name: str = Field(
        default="default_value",
        description="Description of parameter",
    )

    another_param: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Description with validation constraints",
    )

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
```

### `prompts.py` (Optional)

Agent-specific prompt templates and formatting functions.

**Template:**
```python
"""
Prompt templates for the [Agent Name] agent.
"""

from src.agents.shared import format_system_prompt


def get_agent_system_prompt() -> str:
    """
    Get the system prompt for the [Agent Name] agent.

    Returns:
        Formatted system prompt
    """
    return format_system_prompt(
        role="[agent role description]",
        task_description="[detailed task description]",
        guidelines=[
            "guideline 1",
            "guideline 2",
        ],
        constraints=[
            "constraint 1",
            "constraint 2",
        ],
    )


def format_agent_specific_prompt(data: dict) -> str:
    """
    Format an agent-specific prompt with data.

    Args:
        data: Data to include in the prompt

    Returns:
        Formatted prompt string
    """
    # Implement agent-specific formatting
    return f"Custom prompt with {data}"
```

### `tools.py` (Optional)

Agent-specific tool definitions for LLM tool calling.

**Template:**
```python
"""
Tool definitions for the [Agent Name] agent.
"""

from typing import Any, Dict

from src.agents.shared import create_tool_definition


def get_agent_tools() -> list[Dict[str, Any]]:
    """
    Get tool definitions for the [Agent Name] agent.

    Returns:
        List of tool definitions
    """
    return [
        create_tool_definition(
            name="tool_name",
            description="What the tool does",
            parameters={
                "param1": {
                    "type": "string",
                    "description": "Parameter description",
                },
            },
            required=["param1"],
        ),
    ]


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute an agent tool.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        context: Execution context (db, config, etc.)

    Returns:
        Tool execution result
    """
    if tool_name == "tool_name":
        return await _execute_tool_name(arguments, context)

    raise ValueError(f"Unknown tool: {tool_name}")


async def _execute_tool_name(
    arguments: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute specific tool."""
    # Implement tool logic
    return {"result": "success"}
```

## Shared Infrastructure Usage

All agents should leverage the shared infrastructure in `src/agents/shared/`:

### BaseAgent

```python
from src.agents.shared import BaseAgent

class MyAgent(BaseAgent):
    async def execute(self, input_data):
        # Implementation
        pass

    def validate_input(self, input_data):
        # Validation
        return True
```

### Configuration

```python
from src.agents.shared import AgentConfig, ExtractorConfig

# Use base config
config = AgentConfig(model="gpt-4o", temperature=0.7)

# Or use specialized config
config = ExtractorConfig(temperature=0.3)
```

### Prompt Utilities

```python
from src.agents.shared import (
    format_system_prompt,
    format_context_section,
    format_provenance_chain,
    truncate_text,
)

prompt = format_system_prompt(
    role="data processor",
    task_description="Process and analyze data",
)
```

### Tool Utilities

```python
from src.agents.shared import (
    create_tool_definition,
    validate_tool_call,
    extract_tool_arguments,
    format_tool_result,
)

tool = create_tool_definition(
    name="search",
    description="Search for data",
    parameters={"query": {"type": "string"}},
)
```

## Testing

Each agent should have corresponding tests in `tests/agents/[agent_name]/`:

```
tests/
└── agents/
    └── agent_name/
        ├── __init__.py
        ├── test_agent.py          # Main agent tests
        ├── test_config.py         # Configuration tests
        └── test_integration.py    # Integration tests
```

### Test Template

```python
"""Tests for [Agent Name] agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.agent_name import AgentNameAgent


class TestAgentName:
    """Test suite for [Agent Name] agent."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_db):
        """Create agent instance."""
        return AgentNameAgent(db=mock_db)

    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful execution."""
        input_data = {"required_field": "value"}
        result = await agent.execute(input_data)

        assert result["success"] is True

    def test_validate_input_valid(self, agent):
        """Test input validation with valid data."""
        input_data = {"required_field": "value"}
        assert agent.validate_input(input_data) is True

    def test_validate_input_invalid(self, agent):
        """Test input validation with invalid data."""
        input_data = {}

        with pytest.raises(ValueError):
            agent.validate_input(input_data)
```

## Best Practices

1. **Inheritance**: Always inherit from `BaseAgent` for consistency
2. **Type Hints**: Use comprehensive type annotations throughout
3. **Logging**: Use structured logging with `logger.info/debug/error`
4. **Error Handling**: Raise specific exceptions with descriptive messages
5. **Documentation**: Include docstrings for all public methods
6. **Configuration**: Use Pydantic models for type-safe configuration
7. **Testing**: Write comprehensive unit and integration tests
8. **Async**: Use async/await for all I/O operations
9. **Shared Utilities**: Leverage shared infrastructure instead of duplicating code
10. **Naming Conventions**: Use snake_case for functions/variables, PascalCase for classes

## Migration Checklist

When converting an existing agent to this template:

- [ ] Create new directory structure
- [ ] Implement BaseAgent inheritance
- [ ] Move configuration to config.py
- [ ] Extract prompts to prompts.py
- [ ] Extract tools to tools.py
- [ ] Update imports in dependent files
- [ ] Write/update tests
- [ ] Update documentation
- [ ] Verify all functionality works
- [ ] Remove old agent files

## Example Agents

See these agents for reference implementations:

- **Extractor** (`src/agents/extractor/`): Premise extraction and memory formation
- **Dialectic** (`src/agents/dialectic/`): Query answering with context retrieval
- **Dreamer** (`src/agents/dreamer/`): Memory consolidation and improvement

## Questions?

For questions or clarifications about the agent template, see:
- `src/agents/shared/base_agent.py` - BaseAgent implementation
- `AGENT_DEVELOPMENT.md` - Detailed development guidelines (Phase 0B.5)
- `TODO.md` - Implementation plan and progress tracking
