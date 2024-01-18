"""
Below is an implementation of a basic LRUcache that utilizes the built
in OrderedDict data structure.
"""
from collections import OrderedDict
from mediator import SupabaseMediator
import uuid
from typing import List, Dict
from langchain.schema import BaseMessage, Document
from pydantic import BaseModel

# import sentry_sdk


class Conversation:
    "Wrapper Class for storing contexts between channels. Using an object to pass by reference avoid additional cache hits"

    # @sentry_sdk.trace
    def __init__(
        self,
        mediator: SupabaseMediator,
        user_id: str,
        session_id: str = str(uuid.uuid4()),
        metadata: Dict = {},
    ):
        self.mediator: SupabaseMediator = mediator
        self.user_id = user_id
        self.session_id = session_id
        self.metadata: Dict = metadata

    # @sentry_sdk.trace
    def add_message(
        self,
        message_type: str,
        message: BaseMessage,
    ) -> None:
        self.mediator.add_message(self.session_id, message_type, message.content)

    # @sentry_sdk.trace
    def get_messages(self, message_type: str) -> List[BaseMessage]:
        return self.mediator.get_messages(self.session_id, message_type)

    # @sentry_sdk.trace
    def delete(self) -> None:
        self.mediator.delete_session(self.session_id)

    # @sentry_sdk.trace
    # def restart(self) -> None:
    #     self.delete()
    #     representation = self.mediator.add_session(
    #         user_id=self.user_id, location_id=self.location_id
    #     )
    #     self.session_id: str = representation["id"]
    #     self.metadata = representation["metadata"]

    # vector DB fn
    # @sentry_sdk.trace
    def add_texts(self, texts: List[str]) -> None:
        metadatas = [
            {"session_id": self.session_id, "user_id": self.user_id}
            for _ in range(len(texts))
        ]
        self.mediator.vector_table.add_texts(texts, metadatas)

    # vector DB fn
    # @sentry_sdk.trace
    def similarity_search(self, query: str, match_count: int = 5) -> List[Document]:
        return self.mediator.vector_table.similarity_search(
            query=query, k=match_count, filter={"user_id": self.user_id}
        )
