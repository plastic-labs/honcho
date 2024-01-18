import os
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.output_parsers.list import NumberedListOutputParser
from langchain.prompts import (
    load_prompt,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from collections.abc import AsyncIterator
from cache import Conversation
from typing import List


from openai import BadRequestError

import sentry_sdk

load_dotenv()

SYSTEM_THOUGHT = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/thought.yaml")
)
SYSTEM_RESPONSE = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/response.yaml")
)
SYSTEM_THOUGHT_REVISION = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/thought_revision.yaml")
)
SYSTEM_USER_PREDICTION_THOUGHT = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/user_prediction_thought.yaml")
)
SYSTEM_USER_PREDICTION_THOUGHT_REVISION = load_prompt(
    os.path.join(
        os.path.dirname(__file__), "prompts/user_prediction_thought_revision.yaml"
    )
)
SYSTEM_VOE_THOUGHT = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/voe_thought.yaml")
)
SYSTEM_VOE = load_prompt(os.path.join(os.path.dirname(__file__), "prompts/voe.yaml"))
SYSTEM_CHECK_VOE_LIST = load_prompt(
    os.path.join(os.path.dirname(__file__), "prompts/check_voe_list.yaml")
)


class BloomChain:
    "Wrapper class for encapsulating the multiple different chains used in reasoning for the tutor's thoughts"
    llm: AzureChatOpenAI = AzureChatOpenAI(
        deployment_name=os.environ["OPENAI_API_DEPLOYMENT_NAME"],
        temperature=1.2,
        model_kwargs={"top_p": 0.5},
    )
    parser_llm: AzureChatOpenAI = AzureChatOpenAI(
        deployment_name=os.environ["OPENAI_API_DEPLOYMENT_NAME"]
    )
    system_voe_thought: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
        prompt=SYSTEM_VOE_THOUGHT
    )
    system_voe: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
        prompt=SYSTEM_VOE
    )
    system_check_voe_list: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
        prompt=SYSTEM_CHECK_VOE_LIST
    )
    system_thought: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
        prompt=SYSTEM_THOUGHT
    )
    system_thought_revision: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
        prompt=SYSTEM_THOUGHT_REVISION
    )
    system_response: SystemMessagePromptTemplate = SystemMessagePromptTemplate(
        prompt=SYSTEM_RESPONSE
    )
    system_user_prediction_thought: SystemMessagePromptTemplate = (
        SystemMessagePromptTemplate(prompt=SYSTEM_USER_PREDICTION_THOUGHT)
    )
    system_user_prediction_thought_revision: SystemMessagePromptTemplate = (
        SystemMessagePromptTemplate(prompt=SYSTEM_USER_PREDICTION_THOUGHT_REVISION)
    )

    output_parser = NumberedListOutputParser()

    def __init__(self) -> None:
        pass

    @classmethod
    # @sentry_sdk.trace
    def think(cls, cache: Conversation, input: str):
        """Generate Bloom's thought on the user."""
        # load message history
        thought_prompt = ChatPromptTemplate.from_messages(
            [
                cls.system_thought,
                *cache.get_messages("thought"),
                HumanMessage(content=input),
            ]
        )
        chain = thought_prompt | cls.llm


        def save_new_messages(ai_response):
            cache.add_message("thought", HumanMessage(content=input))
            cache.add_message("thought", AIMessage(content=ai_response))
        
        return Streamable(
            chain.astream({}, {"tags": ["thought"], "metadata": {"conversation_id": cache.conversation_id, "user_id": cache.user_id}}),
            save_new_messages 
        )

    @classmethod
    # @sentry_sdk.trace
    def revise_thought(cls, cache: Conversation, input: str, thought: str):
        """Revise Bloom's thought about the user with retrieved personal data"""

        # construct rag prompt, retrieve docs
        query = f"input: {input}\n thought: {thought}"
        docs = cache.similarity_search(query)

        messages = ChatPromptTemplate.from_messages(
            [
                cls.system_thought_revision,
                *cache.get_messages("thought_revision"),
                HumanMessage(content=input),
            ]
        )
        chain = messages | cls.llm

        def save_new_messages(ai_response):
            cache.add_message("thought_revision", HumanMessage(content=input))
            cache.add_message("thought_revision", AIMessage(content=ai_response))

        return Streamable(
            chain.astream({ "thought": thought, "retrieved_vectors": "\n".join(doc.page_content for doc in docs)}, {"tags": ["thought_revision"], "metadata": {"conversation_id": cache.conversation_id, "user_id": cache.user_id}}),
            save_new_messages
        )

    @classmethod
    # @sentry_sdk.trace
    def respond(cls, cache: Conversation, thought: str, input: str):
        """Generate Bloom's response to the user."""
        response_prompt = ChatPromptTemplate.from_messages(
            [
                cls.system_response,
                *cache.get_messages("response"),
                HumanMessage(content=input),
            ]
        )
        chain = response_prompt | cls.llm

        def save_new_messages(ai_response):
            cache.add_message("response", HumanMessage(content=input))
            cache.add_message("response", AIMessage(content=ai_response))

        return Streamable(
            chain.astream({ "thought": thought }, {"tags": ["response"], "metadata": {"conversation_id": cache.conversation_id, "user_id": cache.user_id}}),
            save_new_messages
        )

    @classmethod
    # @sentry_sdk.trace
    async def think_user_prediction(cls, cache: Conversation, input: str):
        """Generate a thought about what the user is going to say"""

        messages = ChatPromptTemplate.from_messages(
            [
                cls.system_user_prediction_thought,
            ]
        )
        chain = messages | cls.llm

        history = unpack_messages(cache.get_messages("response"))

        user_prediction_thought = await chain.ainvoke(
            {"history": history},
            {
                "tags": ["user_prediction_thought"],
                "metadata": {
                    "conversation_id": cache.conversation_id,
                    "user_id": cache.user_id,
                },
            },
        )

        cache.add_message("user_prediction_thought", user_prediction_thought)

        return user_prediction_thought.content

    @classmethod
    # @sentry_sdk.trace
    async def revise_user_prediction_thought(
        cls, cache: Conversation, user_prediction_thought: str, input: str
    ):
        """Revise the thought about what the user is going to say based on retrieval of VoE facts"""

        messages = ChatPromptTemplate.from_messages(
            [
                cls.system_user_prediction_thought_revision,
            ]
        )
        chain = messages | cls.llm

        # construct rag prompt, retrieve docs
        query = f"input: {input}\n thought: {user_prediction_thought}"
        docs = cache.similarity_search(query)

        history = unpack_messages(cache.get_messages("response"))

        user_prediction_thought_revision = await chain.ainvoke(
            {
                "history": history,
                "user_prediction_thought": user_prediction_thought,
                "retrieved_vectors": "\n".join(doc.page_content for doc in docs),
            },
            config={
                "tags": ["user_prediction_thought_revision"],
                "metadata": {
                    "conversation_id": cache.conversation_id,
                    "user_id": cache.user_id,
                },
            },
        )

        cache.add_message(
            "user_prediction_thought_revision", user_prediction_thought_revision
        )

        return user_prediction_thought_revision.content

    @classmethod
    # @sentry_sdk.trace
    async def think_violation_of_expectation(
        cls, cache: Conversation, inp: str, user_prediction_thought_revision: str
    ) -> None:
        """Assess whether expectation was violated, derive and store facts"""

        # format prompt
        messages = ChatPromptTemplate.from_messages([cls.system_voe_thought])
        chain = messages | cls.llm

        voe_thought = await chain.ainvoke(
            {
                "user_prediction_thought_revision": user_prediction_thought_revision,
                "actual": inp,
            },
            config={"tags": ["voe_thought"], "metadata": {"user_id": cache.user_id}},
        )

        cache.add_message("voe_thought", voe_thought)

        return voe_thought.content

    @classmethod
    # @sentry_sdk.trace
    async def violation_of_expectation(
        cls,
        cache: Conversation,
        inp: str,
        user_prediction_thought_revision: str,
        voe_thought: str,
    ) -> None:
        """Assess whether expectation was violated, derive and store facts"""

        # format prompt
        messages = ChatPromptTemplate.from_messages([cls.system_voe])
        chain = messages | cls.llm

        voe = await chain.ainvoke(
            {
                "ai_message": cache.get_messages("response")[-1].content,
                "user_prediction_thought_revision": user_prediction_thought_revision,
                "actual": inp,
                "voe_thought": voe_thought,
            },
            config={"tags": ["voe"], "metadata": {"user_id": cache.user_id}},
        )

        cache.add_message("voe", voe)
        facts = cls.output_parser.parse(voe.content)
        return facts

    @classmethod
    # @sentry_sdk.trace
    async def check_voe_list(cls, cache: Conversation, facts: List[str]):
        """Filter the facts to just new ones"""

        # create the message object from prompt template
        messages = ChatPromptTemplate.from_messages([cls.system_check_voe_list])
        chain = messages | cls.llm

        # unpack the list of strings into one string for similarity search
        # TODO: should we query 1 by 1 and append to an existing facts list?
        query = " ".join(facts)

        # query the vector store
        existing_facts = cache.similarity_search(query, match_count=10)

        filtered_facts = await chain.ainvoke(
            {
                "existing_facts": "\n".join(
                    fact.page_content for fact in existing_facts
                ),
                "facts": "\n".join(fact for fact in facts),
            },
            config={"tags": ["check_voe_list"], "metadata": {"user_id": cache.user_id}},
        )

        data = cls.output_parser.parse(filtered_facts.content)

        # if the check returned "None", write facts to cache
        if not data:
            cache.add_texts(facts)
        else:
            cache.add_texts(data)

    @classmethod
    # @sentry_sdk.trace
    async def chat(cls, cache: Conversation, inp: str) -> tuple[str, str]:
        # VoE has to happen first. If there's user prediction history, derive and store fact(s)
        if cache.get_messages("user_prediction_thought_revision"):
            user_prediction_thought_revision = cache.get_messages(
                "user_prediction_thought_revision"
            )[-1].content

            voe_thought = await cls.think_violation_of_expectation(
                cache, inp, user_prediction_thought_revision
            )
            voe_facts = await cls.violation_of_expectation(
                cache, inp, user_prediction_thought_revision, voe_thought
            )

            if not voe_facts or voe_facts[0] == "None":
                pass
            else:
                await cls.check_voe_list(cache, voe_facts)

        thought_iterator = cls.think(cache, inp)
        thought = await thought_iterator()

        thought_revision_iterator = cls.revise_thought(cache, inp, thought)
        thought_revision = await thought_revision_iterator()

        response_iterator = cls.respond(cache, thought_revision, inp)
        response = await response_iterator()

        user_prediction_thought = await cls.think_user_prediction(cache, inp)
        user_prediction_thought_revision = await cls.revise_user_prediction_thought(
            cache, user_prediction_thought, inp
        )

        return thought, response

    @classmethod
    # @sentry_sdk.trace
    async def stream(cls, cache: Conversation, inp: str):
        # VoE has to happen first. If there's user prediction history, derive and store fact(s)
        try:
            if cache.get_messages("user_prediction_thought_revision"):
                user_prediction_thought_revision = cache.get_messages(
                    "user_prediction_thought_revision"
                )[-1].content

                voe_thought = await cls.think_violation_of_expectation(
                    cache, inp, user_prediction_thought_revision
                )
                voe_facts = await cls.violation_of_expectation(
                    cache, inp, user_prediction_thought_revision, voe_thought
                )

                if not voe_facts or voe_facts[0] == "None":
                    pass
                else:
                    await cls.check_voe_list(cache, voe_facts)

            print("=========================================")
            print("Finished Init")
            print("=========================================")

            thought_iterator = cls.think(cache, inp)
            thought = ""
            async for item in thought_iterator:
                # escape ❀ if present
                item = item.replace("❀", "🌸")
                thought += item
                yield item
            yield "❀"

            print("=========================================")
            print("Finished Thought")
            print("=========================================")

            thought_revision_iterator = cls.revise_thought(cache, inp, thought)
            thought_revision = await thought_revision_iterator()

            response_iterator = cls.respond(cache, thought_revision, inp)
            # response = ""

            async for item in response_iterator:
                # if "❀" in item:
                item = item.replace("❀", "🌸")
                # response += item
                yield item

            print("=========================================")
            print("Finished Response")
            print("=========================================")

            user_prediction_thought = await cls.think_user_prediction(cache, inp)
            user_prediction_thought_revision = await cls.revise_user_prediction_thought(
                cache, user_prediction_thought, inp
            )

            print("=========================================")
            print("Finished User Prediction")
            print("=========================================")
        finally:
            yield "❀"


class Streamable:
    "A async iterator wrapper for langchain streams that saves on completion via callback"

    def __init__(self, iterator: AsyncIterator[BaseMessage], callback):
        self.iterator = iterator
        self.callback = callback
        # self.content: List[Awaitable[BaseMessage]] = []
        self.content = ""

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            data = await self.iterator.__anext__()
            self.content += data.content
            return data.content
        except StopAsyncIteration as e:
            self.callback(self.content)
            raise StopAsyncIteration
        except BadRequestError as e:
            if e.code == "content_filter":
                self.stream_error = True
                self.message = "Sorry, your message was flagged as inappropriate. Please try again."

                return self.message
        except Exception as e:
            raise e

    async def __call__(self):
        async for _ in self:
            pass
        return self.content


def unpack_messages(messages):
    unpacked = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            unpacked += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            unpacked += f"AI: {message.content}\n"
        # Add more conditions here if you're using other message types
    return unpacked
