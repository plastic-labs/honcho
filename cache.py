"""
Below is an implementation of a basic LRUcache that utilizes the built
in OrderedDict data structure.
"""
from collections import OrderedDict
from mediator import SupabaseMediator
import uuid
from typing import List
from langchain.schema import BaseMessage, Document
from pydantic import BaseModel

class Conversation:
    "Wrapper Class for storing contexts between channels. Using an object to pass by reference avoid additional cache hits"
    def __init__(self, mediator: SupabaseMediator, user_id: str, conversation_id: str = str(uuid.uuid4()), location_id: str = "web"):
        self.mediator: SupabaseMediator = mediator
        self.user_id: str = user_id
        self.conversation_id: str = conversation_id 
        self.location_id: str = location_id

    def add_message(self, message_type: str, message: BaseMessage,) -> None:
        self.mediator.add_message(self.conversation_id, self.user_id, message_type, message)

    def messages(self, message_type: str) -> List[BaseMessage]:
        return self.mediator.messages(self.conversation_id, self.user_id, message_type)
    
    # vector DB fn
    def add_texts(self, texts: List[str]) -> None:
        metadatas = [{"conversation_id": self.conversation_id, "user_id": self.user_id} for _ in range(len(texts))]
        self.mediator.vector_table.add_texts(texts, metadatas)

    # vector DB fn
    def similarity_search(self, query: str, match_count: int = 5) -> List[Document]:
        return self.mediator.vector_table.similarity_search(query=query, k=match_count, filter={"user_id": self.user_id})
