from typing import Dict
import requests
import asyncio

from ..architecture import MetacognitionManager, LlmAdapter
from ..architecture.messages import ConversationHistory
from ..user_model import UserModelStorageAdapter, UserRewardModel


class Client:
    def __init__(self, base_url):
        """Constructor for Client"""
        self.base_url = base_url  # Base URL for the instance of the Honcho API

        self.processing_metacognitons = (
            {}
        )  # Dictionary of running metacognitions to be used to check if context is ready

    def get_sessions(self, user_id: str):
        """Return sessions associated with a user

        Args:
            user_id (str): The User ID representing the user, managed by the user

        Returns:
            list[Dict]: List of Session objects

        """
        url = f"{self.base_url}/users/{user_id}/sessions"
        response = requests.get(url)
        return response.json()

    def get_session(self, user_id: str, session_id: int):
        """Get a specific session for a user by ID

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            Dict: The Session object of the requested Session

        """
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}"
        response = requests.get(url)
        return response.json()

    def get_sessions_by_location(self, user_id: str, location_id: str):
        """Get all sessions for a user for a specific location

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session

        Returns:
            list[Dict]: List of Session objects

        """
        url = f"{self.base_url}/users/{user_id}/sessions?location_id={location_id}"
        response = requests.get(url)
        return response.json()

    def create_session(
        self, user_id: str, location_id: str = "default", session_data: Dict = {}
    ):
        """Create a session for a user

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session
            session_data (Dict, optional): Optional session metadata

        Returns:
            Dict: The Session object of the new Session

        """
        data = {"location_id": location_id, "session_data": session_data}
        url = f"{self.base_url}/users/{user_id}/sessions"
        response = requests.post(url, json=data)
        return response.json()

    def update_session(self, user_id: str, session_id: int, session_data: Dict):
        """Update the metadata of a session

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to update
            session_data (Dict): The Session object containing any new metadata

        Returns:
            Dict: The Session object of the updated Session

        """
        data = {"session_data": session_data}
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}"
        response = requests.put(url, json=data)
        return response.json()

    def delete_session(self, user_id: str, session_id: int):
        """Delete a session by marking it as inactive

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to delete

        Returns:
            Dict: A message indicating that the session was deleted

        """
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}"
        response = requests.delete(url)
        return response.json()

    def get_messages_for_session(self, user_id: str, session_id: int):
        """Get all messages for a session

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            list[Dict]: List of Message objects

        """
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}/messages"
        response = requests.get(url)
        return response.json()

    def create_message_for_session(
        self, user_id: str, session_id: int, is_user: bool, content: str
    ):
        """Adds a message to a session

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to add the message to
            is_user (bool): Whether the message is from the user
            content (str): The content of the message

        Returns:
            Dict: The Message object of the added message

        """

        data = {"is_user": is_user, "content": content}
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}/messages"
        response = requests.post(url, json=data)

        # run metacognitive architecture
        if self.metacognitive_architecture_config and response.ok:
            conversation_history = ConversationHistory.from_honcho_dicts(
                self.get_messages_for_session(user_id, session_id)
            )
            user_model = UserRewardModel(
                llm=self.metacognitive_architecture_config["llm"],
                user_model_storage_adapter=self.metacognitive_architecture_config[
                    "user_model_storage_adapter_type"
                ](user_id),
            )
            manager = MetacognitionManager.from_yaml(
                path=self.metacognitive_architecture_config["path"],
                user_model=user_model,
                llm=self.metacognitive_architecture_config["llm"],
                verbose=self.metacognitive_architecture_config["verbose"],
            )

            if is_user:
                task = asyncio.create_task(
                    manager.on_user_message(conversation_history)
                )
            else:
                task = asyncio.create_task(manager.on_ai_message(conversation_history))

            # save running metacognition so we can get context later
            self.processing_metacognitons[user_id] = (manager, task)

        return response.json()

    def register_metacognitive_architecture(
        self,
        path: str,
        user_model_storage_adapter_type: UserModelStorageAdapter,
        llm: LlmAdapter,
        verbose=False,
    ):
        self.metacognitive_architecture_config = {
            "user_model_storage_adapter_type": user_model_storage_adapter_type,
            "path": path,
            "llm": llm,
            "verbose": verbose,
        }

    async def get_context(self, user_id: str):
        if user_id not in self.processing_metacognitons:
            return None

        manager, task = self.processing_metacognitons.get(user_id)
        await task

        context = manager.get_agent_context()
        del self.processing_metacognitons[user_id]

        return context
