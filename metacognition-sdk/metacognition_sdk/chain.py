from langchain.chat_models.base import BaseChatModel
from enum import Enum

from metacognition_sdk.steps import Step
from metacognition_sdk.user_model import UserRewardModel


class Event(Enum):
    OnUserMessage = 0


class Output(Enum):
    Void = 0
    AgentContext = 1


class MetacognitionChain:
    """Defines a metacognitive chain of steps"""

    def __init__(
        self,
        name: str,
        event: Event,
        output: Output,
        steps: list[Step],
        min_completed_turns=0,
        default_output="",
    ):
        self.name = name
        self.event = event
        self.output = output
        self.min_completed_turns = min_completed_turns
        self.default_output = default_output

        self.steps = steps

    async def __call__(self, inputs: dict[str, str]):
        if inputs["turns_completed"] < self.min_completed_turns:
            return self.default_output

        for step in self.steps:
            output = await step(inputs)
            inputs.update({step.name: output})

        last_output = output
        return last_output

    @classmethod
    def from_dict(
        cls,
        name: str,
        chain_dict: dict,
        user_model: UserRewardModel,
        tools: dict,
        llm: BaseChatModel,
    ):
        # load metadata
        event = chain_dict["event"].upper()
        output = chain_dict["output"].upper()
        min_completed_turns = (
            chain_dict["min_completed_turns"]
            if "min_completed_turns" in chain_dict
            else 0
        )
        default_output = (
            chain_dict["default_output"] if "default_output" in chain_dict else ""
        )

        if event == "ON_USER_MESSAGE":
            event = Event.OnUserMessage
        else:
            raise ValueError(f"Unknown event type: {event}")

        if output == "VOID":
            output = Output.Void
        elif output == "AGENT_CONTEXT":
            output = Output.AgentContext
        else:
            raise ValueError(f"Unknown output type: {output}")

        # load steps
        steps = chain_dict["steps"]
        loaded_steps = []

        for step_name, step_dict in steps.items():
            loaded_steps.append(
                Step.from_dict(
                    name=step_name,
                    step_dict=step_dict,
                    llm=llm,
                    user_model=user_model,
                    tools=tools,
                )
            )

        return cls(
            name, event, output, loaded_steps, min_completed_turns, default_output
        )
