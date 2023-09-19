from langchain.memory import PostgresChatMessageHistory
from langchain.schema import Document
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict
from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import uuid
import sentry_sdk
import os
from dotenv import load_dotenv
# Supabase for Postgres Management
from supabase.client import create_client, Client
from typing import List
import json

load_dotenv()

class SupabaseMediator:

    @sentry_sdk.trace
    def __init__(self):
        self.supabase: Client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
        self.memory_table = os.environ["MEMORY_TABLE"]
        self.conversation_table = os.environ["CONVERSATION_TABLE"]
        self.match_function = os.environ["MATCH_FUNCTION"]

        embeddings = OpenAIEmbeddings(
                deployment=os.environ["OPENAI_API_EMBEDDING_NAME"],
                model="text-embedding-ada-002",
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_type=os.environ["OPENAI_API_TYPE"],
                )
        self.vector_table = SupabaseVectorStore(
            embedding=embeddings,
            client=self.supabase,   
            table_name=os.environ["VECTOR_TABLE"], 
            query_name=self.match_function
        )
        # # seed the vector store with facts about bloom
        # seed_docs = [
        #     Document(page_content="Bloom is your learning companion"),
        #     Document(page_content="Bloom can be used for learning just about anything! It's your ultimate school assistant."),
        # ]

        # self.vector_table.add_documents(seed_docs)

    @sentry_sdk.trace
    def messages(self, session_id: str, user_id: str, message_type: str) -> List[BaseMessage]:  # type: ignore
        response = self.supabase.table(self.memory_table).select("message").eq("session_id", session_id).eq("user_id", user_id).eq("message_type", message_type).order("id", desc=True).limit(10).execute()
        items = [record["message"] for record in response.data]
        messages = messages_from_dict(items)
        return messages[::-1]

    @sentry_sdk.trace
    def add_message(self, session_id: str, user_id: str, message_type: str, message: BaseMessage) -> None:
        self.supabase.table(self.memory_table).insert({"session_id": session_id, "user_id": user_id, "message_type": message_type, "message": _message_to_dict(message)}).execute()

    @sentry_sdk.trace
    def conversations(self, location_id: str, user_id: str) -> str | None:
        response = self.supabase.table(self.conversation_table).select("id").eq("location_id", location_id).eq("user_id", user_id).eq("isActive", True).maybe_single().execute()
        if response:
           conversation_id = response.data["id"]
           return conversation_id
        return None
    
    @sentry_sdk.trace
    def add_conversation(self, location_id: str, user_id: str) -> str:
        conversation_id = str(uuid.uuid4())
        self.supabase.table(self.conversation_table).insert({"id": conversation_id, "user_id": user_id, "location_id": location_id}).execute()
        return conversation_id

    @sentry_sdk.trace
    def delete_conversation(self, conversation_id: str) -> None:
        self.supabase.table(self.conversation_table).update({"isActive": False}).eq("id", conversation_id).execute()

