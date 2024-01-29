from abc import ABC, abstractmethod
from typing import Callable, Awaitable

from ..user_model import UserRewardModel
from .interfaces import LlmAdapter


class Step(ABC):
    """
    Abstract class for a step in a Metacognitive Chain.

    Each step receives a dictionary of string inputs and returns a string output
    """

    def __init__(self, name: str):
        """
        Initialize a Step with a given name.

        Args:
            name (str): The name of the step.
        """
        self.name = name

    @abstractmethod
    async def __call__(inputs: dict[str, str]) -> str:
        """
        Abstract method to be implemented by subclasses to perform the step's action.

        Args:
            inputs (dict[str, str]): A dictionary of inputs required for the step.

        Returns:
            str: The result of the step's action.
        """
        return ""

    @classmethod
    def from_dict(
        cls,
        name: str,
        step_dict: dict,
        llm: LlmAdapter,
        user_model: UserRewardModel,
        tools: dict[str, Callable],
    ):
        """
        Factory method to create a Step instance from a dictionary configuration.

        Args:
            name (str): The name of the step.
            step_dict (dict): A dictionary containing the step configuration.
            llm (LlmAdapter): An instance of a language model adapter.
            user_model (UserRewardModel): An instance of the user reward model.
            tools (dict[str, Callable]): A dictionary of callable tools.

        Returns:
            Step: An instance of a subclass of Step based on the provided configuration.
        """
        step_type = step_dict["type"]

        if step_type == "inference":
            return InferenceStep(name=name, llm=llm, prompt=step_dict["prompt"])
        elif step_type == "tool":
            return ToolStep(
                name=name,
                callback=tools[name],
                prompt=step_dict["input"],
            )
        elif step_type == "user_model_revision":
            return ReviseUserModelStep(
                name=name, user_model=user_model, prompt=step_dict["insight"]
            )
        elif step_type == "user_model_query":
            return QueryUserModelStep(
                name=name, user_model=user_model, prompt=step_dict["query"]
            )
        else:
            raise ValueError(f"Unknown step type: {step_type}")


class InferenceStep(Step):
    """
    LLM Inference step for a Metacognitive Chain.

    This step uses a language model to generate a response based on a given prompt
    and the current inputs.
    """

    def __init__(self, name: str, llm: LlmAdapter, prompt: str):
        """
        Initialize an InferenceStep with a name, language model adapter, and prompt.

        Args:
            name (str): The name of the step.
            llm (LlmAdapter): The language model adapter to use for inference.
            prompt (str): The prompt template to use for generating the inference.
        """
        super().__init__(name)

        self.llm = llm
        self.prompt = prompt

    async def __call__(self, inputs: dict[str, str]):
        """
        Perform the inference step using the provided inputs.

        Args:
            inputs (dict[str, str]): A dictionary of inputs required for the step.

        Returns:
            str: The result of the language model inference.
        """

        ai_response = await self.llm.inference(self.prompt.format(**inputs))
        return ai_response


class ToolStep(Step):
    """
    ToolStep for a Metacognitive Chain.

    This step executes a callable tool with the provided inputs and returns its output.
    """

    def __init__(
        self, name: str, callback: Callable[[str], Awaitable[str]], prompt: str
    ):
        """
        Initialize a ToolStep with a name, callback, and prompt.

        Args:
            name (str): The name of the step.
            callback (Callable[[str], Awaitable[str]]): The callable tool to execute.
            prompt (str): The prompt template to use for generating the tool input.
        """
        super().__init__(name)
        self.callback = callback
        self.prompt = prompt

    async def __call__(self, inputs: dict[str, str]) -> str:
        """
        Perform the tool step using the provided inputs.

        Args:
            inputs (dict[str, str]): A dictionary of inputs required for the step.

        Returns:
            str: The result of the tool execution.
        """

        input_str = self.prompt.format(**inputs)
        return await self.callback(input_str)


class ReviseUserModelStep(ToolStep):
    """
    ReviseUserModelStep for a Metacognitive Chain.

    This step is responsible for revising the user model based on an insight.
    """

    def __init__(self, name: str, user_model: UserRewardModel, prompt: str):
        """
        Initialize a ReviseUserModelStep with a name, user model, and prompt.

        Args:
            name (str): The name of the step.
            user_model (UserRewardModel): The user reward model to revise.
            prompt (str): The prompt template to use for generating the insight.
        """

        async def user_model_revision_callback(insight: str):
            await user_model.revise(insight=insight)

            return insight

        super().__init__(name, user_model_revision_callback, prompt)


class QueryUserModelStep(ToolStep):
    """
    QueryUserModelStep for a Metacognitive Chain.

    This step is responsible for querying the user model and returning the result.
    """

    def __init__(self, name: str, user_model: UserRewardModel, prompt: str):
        """
        Initialize a QueryUserModelStep with a name, user model, and prompt.

        Args:
            name (str): The name of the step.
            user_model (UserRewardModel): The user reward model to query.
            prompt (str): The prompt template to use for generating the query.
        """

        # returns the inputted insight for debugging purposes
        async def user_model_query_callback(query: str):
            query_results = await user_model.query(query=query)

            return query_results

        super().__init__(name, user_model_query_callback, prompt)
