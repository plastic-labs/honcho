"""
Below is an implementation of a basic LRUcache that utilizes the built
in OrderedDict data structure.
"""
from collections import OrderedDict
import uuid
from typing import List, Dict

class Session:
    "Wrapper Class for storing contexts between channels. Using an object to pass by reference avoid additional cache hits"
    def __init__(self, user_id: str, session_id: str = str(uuid.uuid4()), location_id: str = "web", metadata: Dict = {}):
        self.user_id: str = user_id
        self.session_id: str = session_id 
        self.location_id: str = location_id
        self.metadata: Dict = metadata

    def add_message(self, message_type: str, message: str,) -> None:
        self.mediator.add_message(self.session_id, self.user_id, message_type, message)

    def messages(self, message_type: str) -> List[str]:
        return self.mediator.messages(self.session_id, self.user_id, message_type)

    def delete(self) -> None:
        self.mediator.delete_session(self.session_id)

    def restart(self) -> None:
        self.delete()
        representation = self.mediator.add_session(user_id=self.user_id, location_id=self.location_id)
        self.session_id: str = representation["id"]
        self.metadata = representation["metadata"]
    
    # vector DB fn
    def add_texts(self, texts: List[str]) -> None:
        metadatas = [{"session_id": self.session_id, "user_id": self.user_id} for _ in range(len(texts))]
        self.mediator.vector_table.add_texts(texts, metadatas)

    # vector DB fn
    def similarity_search(self, query: str, match_count: int = 5) :
        return self.mediator.vector_table.similarity_search(query=query, k=match_count, filter={"user_id": self.user_id})

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None

        # Move the accessed key to the end to indicate it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Session):
        if key in self.cache:
            # If the key already exists, move it to the end and update the value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # If the cache is full, remove the least recently used key-value pair (the first item in the OrderedDict)
                self.cache.popitem(last=False)

        # Add or update the key-value pair at the end of the OrderedDict
        self.cache[key] = value

