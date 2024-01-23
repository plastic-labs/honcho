from abc import ABC, abstractmethod
from typing import Callable, Awaitable
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import SystemMessagePromptTemplate

from metacognition_sdk.user_model import UserRewardModel


class Step(ABC):
    """Abstract class for a step in a Metacognitive Chain"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def __call__(inputs: dict[str, str]) -> str:
        return ""

    @classmethod
    def from_dict(
        cls,
        name: str,
        step_dict: dict,
        llm: BaseChatModel,
        user_model: UserRewardModel,
        tools: dict[str, Callable],
    ):
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
    """LLM Inference step for a Metacognitive Chain"""

    def __init__(self, name: str, llm: BaseChatModel, prompt: str):
        super().__init__(name)

        self.llm = llm
        self.prompt = SystemMessagePromptTemplate.from_template(template=prompt)

    async def __call__(self, inputs: dict[str, str]):
        step_input_vars = self.prompt.input_variables
        filtered_inputs = {key: inputs[key] for key in step_input_vars if key in inputs}

        ai_message = await self.llm.ainvoke([self.prompt.format(**filtered_inputs)])

        return ai_message.content


class ToolStep(Step):
    """ToolStep for a Metacognitive Chain"""

    def __init__(
        self, name: str, callback: Callable[[str], Awaitable[str]], prompt: str
    ):
        super().__init__(name)
        self.callback = callback
        self.prompt = SystemMessagePromptTemplate.from_template(template=prompt)

    async def __call__(self, inputs: dict[str, str]) -> str:
        step_input_vars = self.prompt.input_variables
        filtered_inputs = {key: inputs[key] for key in step_input_vars if key in inputs}

        input_str = self.prompt.format(**filtered_inputs).content
        return await self.callback(input_str)


class ReviseUserModelStep(ToolStep):
    """ReviseUserModelStep for a Metacognitive Chain"""

    def __init__(self, name: str, user_model: UserRewardModel, prompt: str):
        async def user_model_revision_callback(insight: str):
            await user_model.revise(insight=insight)

            return insight

        super().__init__(name, user_model_revision_callback, prompt)


class QueryUserModelStep(ToolStep):
    """QueryUserModelStep for a Metacognitive Chain"""

    def __init__(self, name: str, user_model: UserRewardModel, prompt: str):
        # returns the inputted insight for debugging purposes
        async def user_model_query_callback(query: str):
            query_results = await user_model.query(query=query)

            return query_results

        super().__init__(name, user_model_query_callback, prompt)
