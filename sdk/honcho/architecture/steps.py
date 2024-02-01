from abc import ABC, abstractmethod
from typing import Callable, Awaitable, Union, ClassVar
from pydantic import BaseModel, Field

from ..user_model import UserRewardModel
from .interfaces import LlmAdapter


class BaseStepYamlModel(BaseModel):
    """
    Model for validating the usage of steps in YAML architecture configurations.
    """

    type: ClassVar[str] = "none"


class StepParsingException(Exception):
    """
    Exception for when a step fails to parse.
    """

    def __init__(self, message: str):
        """
        Initialize a StepParsingException with a given message.

        Args:
            message (str): The message of the exception.
        """
        super().__init__(message)
        self.message = message


class Step(ABC):
    """
    Abstract class for a step in a Metacognitive Chain.

    Each step receives a dictionary of string inputs and returns a string output
    """

    step_registry: dict[str, "Step"] = {}

    @classmethod
    def register_step(cls, step: "Step"):
        """
        Register a step to be able to used in architecture YAML configurations.
        """

        cls.step_registry[step.model().type] = step

    @abstractmethod
    def __init__(self, name: str, llm: LlmAdapter, step_definition: BaseStepYamlModel):
        """
        Initialize a Step with a given name.

        Args:
            name (str): The name of the step.
        """
        self.name = name

    @classmethod
    @abstractmethod
    def model() -> BaseStepYamlModel:
        """
        The model of the step to be used to validate it's usage in architecture YAML configurations.
        """
        return BaseStepYamlModel()

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
        # step_type = step_dict["type"]

        # if step_type == "inference":
        #     return InferenceStep(name=name, llm=llm, prompt=step_dict["prompt"])
        # elif step_type == "tool":
        #     return ToolStep(
        #         name=name,
        #         callback=tools[name],
        #         prompt=step_dict["input"],
        #     )
        # elif step_type == "user_model_revision":
        #     return ReviseUserModelStep(
        #         name=name, user_model=user_model, prompt=step_dict["insight"]
        #     )
        # elif step_type == "user_model_query":
        #     return QueryUserModelStep(
        #         name=name, user_model=user_model, prompt=step_dict["query"]
        #     )
        # else:
        #     raise ValueError(f"Unknown step type: {step_type}")

        # Raise exception if step_type is not specified
        if "type" not in step_dict:
            raise StepParsingException(f"Step type not specified: {step_dict}")

        # Get the step type
        step_type = step_dict["type"]

        # Raise exception if step_type is not recognized in Step.step_registry set
        if step_type not in Step.step_registry:
            raise StepParsingException(f"Step type not recognized: {step_type}")

        # Get the step class from the step_registery set
        step_class = Step.step_registry[step_type]

        # Validate the step_dict
        # try:
        step_model = step_class.model().parse_obj(step_dict)

        # initalize with the user model if it's one of the allowed steps that may interact with the user model
        if (
            step_model.type == "user_model_revision"
            or step_model.type == "user_model_query"
        ):
            step = step_class(name, llm, user_model, step_model)
        else:
            step = step_class(name, llm, step_model)

        return step
        # except Exception as e:
        # raise StepParsingException(
        # f"Incorrect definition of '{step_type}' step: '{e}'"
        # ) from e


class InferenceStep(Step):
    """
    LLM Inference step for a Metacognitive Chain.

    This step uses a language model to generate a response based on a given prompt
    and the current inputs.
    """

    class InferenceStepModel(BaseStepYamlModel):
        type: ClassVar[str] = "inference"
        prompt: str = Field(
            ..., description="The prompt template to use when running inference."
        )

    @classmethod
    def model(cls):
        return cls.InferenceStepModel

    def __init__(self, name: str, llm: LlmAdapter, config: InferenceStepModel):
        """
        Initialize an InferenceStep with a name, language model adapter, and prompt.

        Args:
            name (str): The name of the step.
            llm (LlmAdapter): The language model adapter to use for inference.
            config (InferenceStepModel): The configuration for the inference step.
        """
        super().__init__(name, llm, config)

        self.llm = llm
        self.prompt = config.prompt

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

    tools: dict[str, Callable[[str], Awaitable[str]]] = {}

    @classmethod
    def register_tool(self, name: str, tool: Callable[[str], Awaitable[str]]):
        """
        Register a tool to be able to be used in tool steps.
        """

        self.tools[name] = tool

    class ToolStepModel(BaseStepYamlModel):
        type: ClassVar[str] = "tool"
        prompt: str = Field(
            ...,
            description="The prompt template to use for formatting input to the tool.",
        )
        tool: str = Field(
            ...,
            description="The name of the tool to use for executing the tool.",
        )

    @classmethod
    def model(cls):
        return cls.ToolStepModel

    def __init__(self, name: str, llm: LlmAdapter, config: ToolStepModel):
        """
        Initialize a ToolStep with a name, callback, and prompt.

        Args:
            name (str): The name of the step.
            llm (LlmAdapter): The language model to use if needed when executing the tool.
            config (ToolStepModel): The configuration to use for formatting input to the tool.
        """
        super().__init__(name, llm, config)

        # Raise exception if tool isn't registered
        if config.tool not in self.tools:
            raise StepParsingException(f"Callback {config.tool} isn't registered.")

        self.callback = self.tools[config.tool]
        self.prompt = config.prompt

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

    user_model: Union[UserRewardModel, None] = None

    class ReviseUserModelStepModel(BaseStepYamlModel):
        type: ClassVar[str] = "user_model_revision"
        insight: str = Field(
            ...,
            description="Prompt template for generating the insight to revise the user model.",
        )

    @classmethod
    def model(cls):
        return cls.ReviseUserModelStepModel

    def __init__(
        self,
        name: str,
        llm: LlmAdapter,
        user_model: UserRewardModel,
        config: ReviseUserModelStepModel,
    ):
        """
        Initialize a ReviseUserModelStep with a name, user model, and prompt.

        Args:
            name (str): The name of the step.
            llm (LlmAdapter): The language model.
            user_model (UserRewardModel): The user model to use.
            config (ReviseUserModelStepModel): The configuration for the step.
        """

        async def user_model_revision_callback(insight: str):
            await user_model.revise(insight=insight)

            return insight

        ToolStep.register_tool("revise_user_model", user_model_revision_callback)

        tool_step_config = ToolStep.ToolStepModel(
            type=config.type, prompt=config.insight, tool="revise_user_model"
        )
        super().__init__(name, llm, tool_step_config)


class QueryUserModelStep(ToolStep):
    """
    QueryUserModelStep for a Metacognitive Chain.

    This step is responsible for querying the user model and returning the result.
    """

    class QueryUserModelStepModel(BaseStepYamlModel):
        type: ClassVar[str] = "user_model_query"
        query: str = Field(
            ...,
            description="Prompt template for generating the query for the user model.",
        )

    @classmethod
    def model(cls):
        return cls.QueryUserModelStepModel

    def __init__(
        self,
        name: str,
        llm: LlmAdapter,
        user_model: UserRewardModel,
        config: QueryUserModelStepModel,
    ):
        """
        Initialize a QueryUserModelStep with a name, user model, and prompt.

        Args:
            name (str): The name of the step.
            llm (LlmAdapter): The LlmAdapter to use.
            user_model (UserRewardModel): The user model to use.
            config (QueryUserModelStepModel): The configuration for the step.
        """

        # returns the inputted insight for debugging purposes
        async def user_model_query_callback(query: str):
            query_results = await user_model.query(query=query)

            return query_results

        ToolStep.register_tool("query_user_model", user_model_query_callback)

        tool_step_config = ToolStep.ToolStepModel(
            type=config.type, prompt=config.query, tool="query_user_model"
        )
        super().__init__(name, llm, tool_step_config)


# register default steps
Step.register_step(InferenceStep)
Step.register_step(ReviseUserModelStep)
Step.register_step(QueryUserModelStep)
