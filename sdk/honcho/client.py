from typing import Dict
import requests

class Client:
    def __init__(self, base_url):
        """Constructor for Client"""
        self.base_url = base_url # Base URL for the instance of the Honcho API

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

    def create_session(self, user_id: str, location_id: str = "default", session_data: Dict = {}):
        """Create a session for a user

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session
            session_data (Dict, optional): Optional session metadata

        Returns:
            Dict: The Session object of the new Session

        """
        data = {
                "location_id": location_id,
                "session_data": session_data
                }
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
        data = {
                "session_data": session_data
                }
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

    def create_message_for_session(self, user_id: str, session_id: int, is_user: bool, content: str):
        """Adds a message to a session
        
        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to add the message to
            is_user (bool): Whether the message is from the user
            content (str): The content of the message

        Returns:
            Dict: The Message object of the added message

        """
        data = {
                "is_user": is_user,
                "content": content
                }
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}/messages"
        response = requests.post(url, json=data)
        return response.json()

