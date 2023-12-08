# from langchain.memory import PostgresChatMessageHistory
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
from typing import List, Dict
import json

load_dotenv()

class SupabaseMediator:

    @sentry_sdk.trace
    def __init__(self):
        self.supabase: Client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
        self.memory_table = os.environ["MEMORY_TABLE"]
        self.session_table = os.environ["SESSION_TABLE"]
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

    # @sentry_sdk.trace
    # def conversations(self, location_id: str, user_id: str) -> str | None:
    #     response = self.supabase.table(self.session_table).select("id").eq("location_id", location_id).eq("user_id", user_id).eq("isActive", True).maybe_single().execute()
    #     if response:
    #        session_id = response.data["id"]
    #        return session_id
    #     return None

    

    # CRUD for sessions    
        
    @sentry_sdk.trace
    def add_session(self, location_id: str, user_id: str, metadata: Dict = {}) -> Dict:
        session_id = str(uuid.uuid4())
        payload = {
                "id": session_id, 
                "user_id": user_id, 
                "location_id": location_id,
                "metadata": metadata,
        }
        representation = self.supabase.table(self.session_table).insert(payload, returning="representation").execute() # type: ignore
        print("========================================")
        print(representation)
        print("========================================")
        return representation.data[0]
        # self.supabase.table(self.session_table).insert({"id": session_id, "user_id": user_id, "location_id": location_id}).execute()

    @sentry_sdk.trace
    def session(self, session_id: str) -> Dict | None:
        response = self.supabase.table(self.session_table).select("*").eq("id", session_id).eq("isActive", True).maybe_single().execute()
        if response:
           return response.data
        return None

    @sentry_sdk.trace
    def sessions(self, location_id: str, user_id: str, single: bool = True) -> List[Dict] | None:
        try:
            response = self.supabase.table(self.session_table).select(*["id", "metadata"], count="exact").eq("location_id", location_id).eq("user_id", user_id).eq("isActive", True).order("created_at", desc=True).execute()
            if response is not None and response.count is not None:
                if (response.count > 1) and single:
                    # If there is more than 1 active session mark the rest for deletion
                    session_ids = [record["id"] for record in response.data[1:]]
                    self._cleanup_sessions(session_ids) # type: ignore
                    return [response.data[0]]
                else:
                    return response.data
            return None
        except Exception as e:
            print("========================================")
            print(e)
            print("========================================")
            return None

    @sentry_sdk.trace
    def update_session(self, user_id: str, session_id: str, metadata: Dict) -> None:
       cur =  self.supabase.table(self.session_table).select("metadata").eq("id", session_id).eq("user_id", user_id).single().execute()
       if cur.data['metadata'] is not None:
           new_metadata = cur.data['metadata'].copy()
           new_metadata.update(metadata)
       else:
           new_metadata = metadata
       self.supabase.table(self.session_table).update({"metadata": new_metadata}, returning="representation").eq("id", session_id).execute() # type: ignore

    @sentry_sdk.trace
    def delete_session(self, user_id: str, session_id: str) -> None:
        self.supabase.table(self.session_table).update({"isActive": False}).eq("id", session_id).eq("user_id", user_id).execute()
    
    # Session Helper Methods

    @sentry_sdk.trace
    def _cleanup_sessions(self, session_ids: List[str]) -> None:
        for session_id in session_ids:
            self.supabase.table(self.session_table).update({"isActive": False}).eq("id", session_id).execute()

    # Session Usage Read & Write

    @sentry_sdk.trace
    def messages(self, session_id: str, user_id: str, message_type: str) -> List[BaseMessage]:  # type: ignore
        response = self.supabase.table(self.memory_table).select("message").eq("session_id", session_id).eq("user_id", user_id).eq("message_type", message_type).order("id", desc=True).limit(10).execute()
        items = [record["message"] for record in response.data]
        messages = messages_from_dict(items)
        return messages[::-1]

    @sentry_sdk.trace
    def add_message(self, session_id: str, user_id: str, message_type: str, message: BaseMessage) -> None:
        payload =  {
                 "session_id": session_id, 
                 "user_id": user_id, 
                 "message_type": message_type, 
                 "message": _message_to_dict(message)
                }
        self.supabase.table(self.memory_table).insert(payload).execute()
