"""
Tests for BaseAgent implementation.

Test criteria:
- TC-0B.1: BaseAgent can be instantiated (with mock abstract methods)
- TC-0B.2: BaseAgent methods work correctly (run, pre_execute, post_execute, trace_execution)
"""

import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, AsyncMock

from src.agents.shared import BaseAgent


class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(self, db, config=None, **kwargs):
        super().__init__(db, config, **kwargs)
        self.execution_log = []

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock execute implementation."""
        self.execution_log.append("execute")
        return {
            "result": "success",
            "processed": input_data.get("data", ""),
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Mock validate_input implementation."""
        self.execution_log.append("validate_input")
        return "data" in input_data


class TestBaseAgent:
    """Test suite for BaseAgent functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = MagicMock()
        return db

    @pytest.fixture
    def test_agent(self, mock_db):
        """Create TestAgent instance."""
        return TestAgent(db=mock_db, config=None)

    def test_agent_initialization(self, mock_db):
        """TC-0B.1: BaseAgent can be instantiated."""
        agent = TestAgent(db=mock_db, config=None)

        assert agent.db == mock_db
        assert agent.config is None
        assert agent.agent_type == "testagent"
        assert hasattr(agent, "execution_log")

    def test_agent_initialization_with_config(self, mock_db):
        """TC-0B.1: BaseAgent can be instantiated with config."""
        from src.agents.shared import AgentConfig

        config = AgentConfig(
            model="gpt-4o-mini",
            temperature=0.5,
            timeout=30,
        )

        agent = TestAgent(db=mock_db, config=config)

        assert agent.config == config
        assert agent.config.model == "gpt-4o-mini"
        assert agent.config.temperature == 0.5

    def test_agent_initialization_with_kwargs(self, mock_db):
        """TC-0B.1: BaseAgent can be instantiated with additional kwargs."""
        agent = TestAgent(
            db=mock_db,
            config=None,
            custom_param="test_value",
            another_param=42,
        )

        assert agent.custom_param == "test_value"
        assert agent.another_param == 42

    @pytest.mark.asyncio
    async def test_execute_method(self, test_agent):
        """TC-0B.2: Execute method works correctly."""
        input_data = {"data": "test input"}
        output = await test_agent.execute(input_data)

        assert output["result"] == "success"
        assert output["processed"] == "test input"
        assert "execute" in test_agent.execution_log

    def test_validate_input_method(self, test_agent):
        """TC-0B.2: Validate input method works correctly."""
        valid_input = {"data": "test"}
        invalid_input = {"wrong_key": "test"}

        assert test_agent.validate_input(valid_input) is True
        assert test_agent.validate_input(invalid_input) is False
        assert test_agent.execution_log.count("validate_input") == 2

    @pytest.mark.asyncio
    async def test_pre_execute_hook(self, test_agent):
        """TC-0B.2: Pre-execute hook validates input."""
        valid_input = {"data": "test"}
        result = await test_agent.pre_execute(valid_input)

        assert result == valid_input
        assert "validate_input" in test_agent.execution_log

    @pytest.mark.asyncio
    async def test_pre_execute_hook_invalid_input(self, test_agent):
        """TC-0B.2: Pre-execute hook raises error for invalid input."""
        invalid_input = {"wrong_key": "test"}

        with pytest.raises(ValueError, match="Invalid input for testagent agent"):
            await test_agent.pre_execute(invalid_input)

    @pytest.mark.asyncio
    async def test_post_execute_hook(self, test_agent, mock_db):
        """TC-0B.2: Post-execute hook traces execution."""
        input_data = {"data": "test"}
        output = {"result": "success"}

        result = await test_agent.post_execute(input_data, output)

        assert result == output

    @pytest.mark.asyncio
    async def test_trace_execution(self, test_agent):
        """TC-0B.2: Trace execution logs correctly."""
        input_data = {"data": "test"}
        output = {"result": "success"}
        metadata = {"execution_time": 1.5}

        # Should not raise any errors
        await test_agent.trace_execution(input_data, output, metadata)

    @pytest.mark.asyncio
    async def test_run_pipeline_success(self, test_agent):
        """TC-0B.2: Full run pipeline executes correctly."""
        input_data = {"data": "test input"}

        # Clear execution log
        test_agent.execution_log = []

        output = await test_agent.run(input_data)

        assert output["result"] == "success"
        assert output["processed"] == "test input"

        # Verify execution order
        assert "validate_input" in test_agent.execution_log
        assert "execute" in test_agent.execution_log
        assert test_agent.execution_log.index("validate_input") < test_agent.execution_log.index("execute")

    @pytest.mark.asyncio
    async def test_run_pipeline_validation_failure(self, test_agent):
        """TC-0B.2: Run pipeline fails on invalid input."""
        invalid_input = {"wrong_key": "test"}

        with pytest.raises(ValueError, match="Invalid input for testagent agent"):
            await test_agent.run(invalid_input)

    @pytest.mark.asyncio
    async def test_run_pipeline_execution_failure(self, mock_db):
        """TC-0B.2: Run pipeline handles execution errors."""

        class FailingAgent(BaseAgent):
            async def execute(self, input_data):
                raise RuntimeError("Execution failed")

            def validate_input(self, input_data):
                return True

        agent = FailingAgent(db=mock_db)

        with pytest.raises(RuntimeError, match="Execution failed"):
            await agent.run({"data": "test"})

    def test_agent_repr(self, test_agent):
        """Test agent string representation."""
        repr_str = repr(test_agent)
        assert "TestAgent" in repr_str
        assert "testagent" in repr_str
