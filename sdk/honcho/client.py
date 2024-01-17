import requests

from dotenv import load_dotenv
load_dotenv()


class Client:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_sessions(self, user_id):
        url = f"{self.base_url}/"
        pass

    def get_session(self, user_id, session_id):
        url = f"{self.base_url}/"
        pass

    def create_session(self, user_id, location_id):
        url = f"{self.base_url}/"
        pass

    def update_session(self, user_id, session_id):
        url = f"{self.base_url}/"
        pass

    def delete_session(self, user_id, session_id):
        url = f"{self.base_url}/"
        pass

    def get_messages_for_session(self, user_id, session_id):
        url = f"{self.base_url}/"
        pass

    def create_message_for_session(self, user_id, session_id):
        url = f"{self.base_url}/"
        pass

    def get_data(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params)
        return response.json()





