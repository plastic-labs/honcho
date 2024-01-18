from typing import Dict
import requests

class Client:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_sessions(self, user_id: str):
        url = f"{self.base_url}/users/{user_id}/sessions"
        response = requests.get(url)
        return response.json()

    def get_session(self, user_id: str, session_id: int):
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}"
        response = requests.get(url)
        return response.json()

    def get_sessions_by_location(self, user_id: str, location_id: str):
        url = f"{self.base_url}/users/{user_id}/sessions?location_id={location_id}"
        response = requests.get(url)
        return response.json()

    def create_session(self, user_id: str, location_id: str = "default", session_data: Dict = {}):
        data = {
                "location_id": location_id,
                "session_data": session_data
                }
        url = f"{self.base_url}/users/{user_id}/sessions"
        response = requests.post(url, json=data)
        return response.json()

    def update_session(self, user_id: str, session_id: int, session_data: Dict):
        data = {
                "session_data": session_data
                }
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}"
        response = requests.put(url, json=data)
        return response.json()

    def delete_session(self, user_id: str, session_id: int):
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}"
        response = requests.delete(url)
        return response.json()

    def get_messages_for_session(self, user_id: str, session_id: int):
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}/messages"
        response = requests.get(url)
        return response.json()

    def create_message_for_session(self, user_id: str, session_id: int, is_user: bool, content: str):
        data = {
                "is_user": is_user,
                "content": content
                }
        url = f"{self.base_url}/users/{user_id}/sessions/{session_id}/messages"
        response = requests.post(url, json=data)
        return response.json()

