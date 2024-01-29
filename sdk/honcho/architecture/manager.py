import yaml

from .chain import (
    MetacognitionChain,
    Event,
    Output,
)
from ..user_model import UserRewardModel
from .interfaces import LlmAdapter
from .messages import ConversationHistory

from tqdm.rich import tqdm
from rich import print

# ignore experimental warning for tqdm.rich
from tqdm import TqdmExperimentalWarning
import warnings

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class MetacognitionManager:
    def __init__(
        self,
        metacognition_dict: dict,
        user_model: UserRewardModel,
        llm: LlmAdapter,
        tools={},
        default_agent_context=None,
        verbose=False,
    ):
        self.chains = [[] for event in Event]

        self.user_model = user_model
        self.llm = llm
        self.tools = tools
        self.verbose = verbose

        for chain_name in metacognition_dict["metacognition"]["chains"]:
            chain_dict = metacognition_dict["metacognition"]["chains"][chain_name]

            self.register_metacognition_chain(chain_name, chain_dict)

        self.agent_context = None
        self.default_agent_context = default_agent_context

    @classmethod
    def from_yaml(
        cls,
        path: str,
        user_model: UserRewardModel,
        llm: LlmAdapter,
        tools={},
        verbose=False,
    ):
        with open(path, "r") as file:
            return cls(
                metacognition_dict=yaml.safe_load(file),
                user_model=user_model,
                llm=llm,
                tools=tools,
                verbose=verbose,
            )

    def register_metacognition_chain(self, chain_name: str, chain_dict: dict):
        chain = MetacognitionChain.from_dict(
            chain_name,
            chain_dict,
            self.user_model,
            self.tools,
            self.llm,
        )

        self.chains[chain.event.value].append(chain)

    async def on_user_message(self, conversation_history: ConversationHistory):
        on_user_message_chains = self.chains[Event.OnUserMessage.value]

        conversation_history_str = str(conversation_history)
        prev_conversation_history_str = str(conversation_history[:-1])
        user_message = conversation_history[-1]["content"]
        turns_completed = (len(conversation_history.messages) - 1) // 2

        inputs = {
            "conversation_history": conversation_history_str,
            "prev_conversation_history": prev_conversation_history_str,
            "user_message": user_message,
            "turns_completed": turns_completed,
        }

        await self.execute_chains(
            on_user_message_chains, inputs, event="on_user_message"
        )

    async def execute_chains(self, chains, inputs, event=""):
        with tqdm(
            chains,
            desc=f"Executing metacognitive chains for '{event}'",
            dynamic_ncols=True,
            disable=not self.verbose,
        ) as pbar:
            if self.verbose:
                print(
                    "[bold white]========================================[/bold white]"
                )

            for chain in pbar:
                result = await chain(inputs)

                if chain.output == Output.AgentContext:
                    self.agent_context = result

                if self.verbose:
                    print(f"[bold blue]Chain: '{chain.name}'[/bold blue]")
                    print(
                        f"[bold green]Output: {result if result.strip() else 'None'}[/bold green]"
                    )
                    print(
                        "[bold white]----------------------------------------[/bold white]"
                    )

                pbar.update()

            if self.verbose:
                print(
                    f"[bold yellow]User Model: {await self.user_model.user_model_storage_adapter.get_user_model()}[/bold yellow]"
                )
                print(
                    "[bold white]========================================[/bold white]"
                )

    async def on_ai_message(self, conversation_history: ConversationHistory):
        pass

    def get_agent_context(self):
        agent_context = (
            self.agent_context
            if self.agent_context is not None
            else self.default_agent_context
        )

        self.agent_context = self.default_agent_context

        return agent_context
