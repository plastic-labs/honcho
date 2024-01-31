import json
from typing import Dict
import requests


class Client:
    def __init__(self, app_id: str, base_url: str = "https://demo.honcho.dev"):
        """Constructor for Client"""
        self.base_url = base_url  # Base URL for the instance of the Honcho API
        self.app_id = app_id # Representing ID of the client application

    @property
    def common_prefix(self):
        return f"{self.base_url}/apps/{self.app_id}"

    def get_session(self, user_id: str, session_id: int):
        """Get a specific session for a user by ID

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            Dict: The Session object of the requested Session

        """
        url = f"{self.common_prefix}/users/{user_id}/sessions/{session_id}"
        response = requests.get(url)
        data = response.json()
        return Session(
            client=self,
            id=data["id"],
            user_id=data["user_id"],
            location_id=data["location_id"],
            is_active=data["is_active"],
            session_data=data["session_data"],
        )

    def get_sessions(self, user_id: str, location_id: str | None = None):
        """Return sessions associated with a user

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session

        Returns:
            list[Dict]: List of Session objects

        """
        url = f"{self.common_prefix}/users/{user_id}/sessions" + (
            f"?location_id={location_id}" if location_id else ""
        )
        response = requests.get(url)
        return [
            Session(
                client=self,
                id=session["id"],
                user_id=session["user_id"],
                location_id=session["location_id"],
                is_active=session["is_active"],
                session_data=session["session_data"],
            )
            for session in response.json()
        ]

    def create_session(
        self, user_id: str, location_id: str = "default", session_data: Dict = {}
    ):
        """Create a session for a user

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session
            session_data (Dict, optional): Optional session metadata

        Returns:
            Dict: The Session object of the new Session`

        """
        data = {"location_id": location_id, "session_data": session_data}
        url = f"{self.common_prefix}/users/{user_id}/sessions"
        response = requests.post(url, json=data)
        data = response.json()
        return Session(
            self,
            id=data["id"],
            user_id=user_id,
            location_id=location_id,
            session_data=session_data,
            is_active=data["is_active"],
        )


class Session:
    def __init__(
        self,
        client: Client,
        id: int,
        user_id: str,
        location_id: str,
        session_data: dict | str,
        is_active: bool,
    ):
        """Constructor for Session"""
        self.base_url = client.base_url
        self.app_id = client.app_id
        self.id = id
        self.user_id = user_id
        self.location_id = location_id
        self.session_data = (
            session_data if isinstance(session_data, dict) else json.loads(session_data)
        )
        self._is_active = is_active

    @property
    def common_prefix(self):
        return f"{self.base_url}/apps/{self.app_id}"

    def __str__(self):
        return f"Session(id={self.id}, app_id={self.app_id}, user_id={self.user_id}, location_id={self.location_id}, session_data={self.session_data}, is_active={self.is_active})"

    @property
    def is_active(self):
        return self._is_active

    def create_message(self, is_user: bool, content: str):
        """Adds a message to the session

        Args:
            is_user (bool): Whether the message is from the user
            content (str): The content of the message

        Returns:
            Dict: The Message object of the added message

        """
        if not self.is_active:
            raise Exception("Session is inactive")
        data = {"is_user": is_user, "content": content}
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages"
        response = requests.post(url, json=data)
        data = response.json()
        return Message(self, id=data["id"], is_user=is_user, content=content)

    def get_messages(self):
        """Get all messages for a session

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            list[Dict]: List of Message objects

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages"
        response = requests.get(url)
        data = response.json()
        return [
            Message(
                self,
                id=message["id"],
                is_user=message["is_user"],
                content=message["content"],
            )
            for message in data
        ]

    def update(self, session_data: Dict):
        """Update the metadata of a session

        Args:
            session_data (Dict): The Session object containing any new metadata


        Returns:
            boolean: Whether the session was successfully updated
        """
        info = {"session_data": session_data}
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}"
        response = requests.put(url, json=info)
        success = response.status_code < 400
        self.session_data = session_data
        return success

    def delete(self):
        """Delete a session by marking it as inactive"""
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}"
        response = requests.delete(url)
        self._is_active = False


class Message:
    def __init__(self, session: Session, id: int, is_user: bool, content: str):
        """Constructor for Message"""
        self.session = session
        self.id = id
        self.is_user = is_user
        self.content = content

    def __str__(self):
        return f"Message(id={self.id}, is_user={self.is_user}, content={self.content})"
