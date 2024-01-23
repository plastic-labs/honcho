from langchain.memory import ChatMessageHistory
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import BaseMessage
import yaml

from metacognition_sdk.chain import (
    MetacognitionChain,
    Event,
    Output,
)
from metacognition_sdk.user_model import UserRewardModel

from tqdm.rich import tqdm
from rich import print

# ignore experimental warning for tqdm.rich
from tqdm import TqdmExperimentalWarning
import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def conversation_history_to_str(conversation_history: list[BaseMessage]):
    """Return conversation history as a string formatted as: User: ...\nAI:...\netc..."""
    conversation_str = ""

    for message in conversation_history:
        speaker = message.type
        conversation_str += f"{speaker}: {message.content}\n"

    return conversation_str.strip()


class MetacognitionManager:
    def __init__(
        self,
        metacognition_dict: dict,
        user_model: UserRewardModel,
        llm: BaseChatModel,
        tools={},
        default_agent_context=None,
    ):
        self.chains = [[] for event in Event]

        self.user_model = user_model
        self.llm = llm
        self.tools = tools

        for chain_name in metacognition_dict["metacognition"]["chains"]:
            chain_dict = metacognition_dict["metacognition"]["chains"][chain_name]

            self.register_metacognition_chain(chain_name, chain_dict)

        self.agent_context = None
        self.default_agent_context = default_agent_context

    @classmethod
    def from_yaml(
        cls, path: str, user_model: UserRewardModel, llm: BaseChatModel, tools={}
    ):
        with open(path, "r") as file:
            return cls(yaml.safe_load(file), user_model, llm, tools)

    def register_metacognition_chain(self, chain_name: str, chain_dict: dict):
        chain = MetacognitionChain.from_dict(
            chain_name, chain_dict, self.user_model, self.tools, self.llm
        )

        self.chains[chain.event.value].append(chain)

    async def on_user_message(
        self, conversation_history: ChatMessageHistory, verbose=False
    ):
        on_user_message_chains = self.chains[Event.OnUserMessage.value]

        conversation_history_str = conversation_history_to_str(
            conversation_history.messages
        )
        prev_conversation_history_str = conversation_history_to_str(
            conversation_history.messages[:-1]
        )
        user_message = conversation_history.messages[-1].content
        turns_completed = (len(conversation_history.messages) - 1) // 2

        inputs = {
            "conversation_history": conversation_history_str,
            "prev_conversation_history": prev_conversation_history_str,
            "user_message": user_message,
            "turns_completed": turns_completed,
        }

        await self.execute_chains(
            on_user_message_chains, inputs, verbose, event="on_user_message"
        )

    async def execute_chains(self, chains, inputs, verbose=False, event=""):
        with tqdm(
            chains,
            desc=f"Executing metacognitive chains for '{event}'",
            dynamic_ncols=True,
            disable=not verbose,
        ) as pbar:
            if verbose:
                print(
                    "[bold white]========================================[/bold white]"
                )

            for chain in pbar:
                result = await chain(inputs)

                if chain.output == Output.AgentContext:
                    self.agent_context = result

                if verbose:
                    print(f"[bold blue]Chain: '{chain.name}'[/bold blue]")
                    print(f"[bold green]Output: {result}[/bold green]")
                    print(
                        "[bold white]----------------------------------------[/bold white]"
                    )

                pbar.update()

            if verbose:
                print(
                    f"[bold yellow]User Model: {self.user_model.user_context}[/bold yellow]"
                )
                print(
                    "[bold white]========================================[/bold white]"
                )

    async def on_ai_message(self, conversation_history: ChatMessageHistory):
        pass

    def get_agent_context(self):
        agent_context = (
            self.agent_context
            if self.agent_context is not None
            else self.default_agent_context
        )

        self.agent_context = self.default_agent_context

        return agent_context
