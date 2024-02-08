import json
import uuid
from typing import Dict, Optional 
import httpx
from .schemas import Message, Metamessage

class GetPage:
    """Base class for receiving Paginated API results"""
    def __init__(self, response: Dict) -> None:
        """Constructor for Page with relevant information about the results and pages

        Args:
            response (Dict): Response from API with pagination information
        """
        self.total = response["total"]
        self.page = response["page"]
        self.page_size = response["size"]
        self.pages = response["pages"]
        self.items =[]

    def next(self):
        """Shortcut method to Get the next page of results"""
        pass

class GetSessionPage(GetPage):
    """Paginated Results for Get Session Requests"""

    def __init__(self, client, options: Dict, response: Dict):
        """Constructor for Page Result from Session Get Request
        
        Args:
            client (Client): Honcho Client
            options (Dict): Options for the request used mainly for next() to filter queries. The two parameters available are user_id which is required and location_id which is optional
            response (Dict): Response from API with pagination information
        """
        super().__init__(response)
        self.client = client
        self.user_id = options["user_id"]
        self.location_id = options["location_id"]
        self.items = [
            Session(
                client=client,
                id=session["id"],
                user_id=session["user_id"],
                location_id=session["location_id"],
                is_active=session["is_active"],
                session_data=session["session_data"],
            )
            for session in response["items"]
        ]
       
    def next(self):
        """Get the next page of results
        Returns:
            GetSessionPage | None: Next Page of Results or None if there are no more sessions to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return self.client.get_sessions(self.user_id, self.location_id, self.page + 1, self.page_size)

class GetMessagePage(GetPage):
    """Paginated Results for Get Session Requests"""

    def __init__(self, session, response: Dict):
        """Constructor for Page Result from Session Get Request
        
        Args:
            session (Session): Session the returned messages are associated with
            response (Dict): Response from API with pagination information
        """
        super().__init__(response)
        self.session = session
        self.items = [
                Message(
                session_id=session.id,
                id=message["id"],
                is_user=message["is_user"],
                content=message["content"],
                created_at=message["created_at"],
            )
            for message in response["items"]
        ]

    def next(self):
        """Get the next page of results
        Returns:
            GetMessagePage | None: Next Page of Results or None if there are no more messages to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return self.session.get_messages((self.page + 1), self.page_size)

class GetMetamessagePage(GetPage):
    
    def __init__(self, session, options: Dict, response: Dict) -> None:
        """Constructor for Page Result from Metamessage Get Request
        
        Args:
            session (Session): Session the returned messages are associated with
            options (Dict): Options for the request used mainly for next() to filter queries. The two parameters available are message_id and metamessage_type which are both required
            response (Dict): Response from API with pagination information
        """
        super().__init__(response)
        self.session = session
        self.message_id = options["message_id"]
        self.metamessage_type = options["metamessage_type"]
        self.items = [
                Metamessage(
                    id=metamessage["id"],
                    message_id=metamessage["message_id"],
                    metamessage_type=metamessage["metamessage_type"],
                    content=metamessage["content"],
                    created_at=metamessage["created_at"],
                    )
                for metamessage in response["items"]
        ]

    def next(self):
        """Get the next page of results
        Returns:
            GetMetamessagePage | None: Next Page of Results or None if there are no more metamessages to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return self.session.get_metamessages(metamessage_type=self.metamessage_type, message=self.message_id, page=(self.page + 1), page_size=self.page_size)



class Client:
    """Honcho API Client Object"""

    def __init__(self, app_id: str, base_url: str = "https://demo.honcho.dev"):
        """Constructor for Client"""
        self.base_url = base_url  # Base URL for the instance of the Honcho API
        self.app_id = app_id # Representing ID of the client application
        self.client = httpx.Client()

    @property
    def common_prefix(self):
        """Shorcut for common API prefix. made a property to prevent tampering"""
        return f"{self.base_url}/apps/{self.app_id}"

    def get_session(self, user_id: str, session_id: uuid.UUID):
        """Get a specific session for a user by ID

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (uuid.UUID): The ID of the Session to retrieve

        Returns:
            Session: The Session object of the requested Session

        """
        url = f"{self.common_prefix}/users/{user_id}/sessions/{session_id}"
        response = self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Session(
            client=self,
            id=data["id"],
            user_id=data["user_id"],
            location_id=data["location_id"],
            is_active=data["is_active"],
            session_data=data["session_data"],
        )

    def get_sessions(self, user_id: str, location_id: Optional[str] = None, page: int = 1, page_size: int = 50):
        """Return sessions associated with a user paginated

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return

        Returns:
            GetSessionPage: Page or results for get_sessions query

        """
        url = f"{self.common_prefix}/users/{user_id}/sessions?page={page}&size={page_size}" + (
            f"&location_id={location_id}" if location_id else ""
        )
        response = self.client.get(url)
        response.raise_for_status()
        data = response.json()
        options = {
                "location_id": location_id,
                "user_id": user_id
                }
        return GetSessionPage(self, options, data)

    def get_sessions_generator(self, user_id: str, location_id: Optional[str] = None):
        """Shortcut Generator for get_sessions. Generator to iterate through all sessions for a user in an app

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session

        Yields:
            Session: The Session object of the requested Session

        """
        page = 1
        page_size = 50
        get_session_response = self.get_sessions(user_id, location_id, page, page_size)
        while True:
            # get_session_response = self.get_sessions(user_id, location_id, page, page_size)
            for session in get_session_response.items:
                yield session

            new_sessions = get_session_response.next()
            if not new_sessions:
                break
           
            get_session_response = new_sessions

    def create_session(
        self, user_id: str, location_id: str = "default", session_data: Dict = {}
    ):
        """Create a session for a user

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session
            session_data (Dict, optional): Optional session metadata

        Returns:
            Session: The Session object of the new Session

        """
        data = {"location_id": location_id, "session_data": session_data}
        url = f"{self.common_prefix}/users/{user_id}/sessions"
        response = self.client.post(url, json=data)
        response.raise_for_status()
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
    """Represents a single session for a user in an app"""

    def __init__(
        self,
        client: Client,
        id: uuid.UUID,
        user_id: str,
        location_id: str,
        session_data: dict | str,
        is_active: bool,
    ):
        """Constructor for Session"""
        self.base_url = client.base_url
        self.client = client.client
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
        """Shortcut for common API prefix. made a property to prevent tampering"""
        return f"{self.base_url}/apps/{self.app_id}"

    def __str__(self):
        """String representation of Session"""
        return f"Session(id={self.id}, app_id={self.app_id}, user_id={self.user_id}, location_id={self.location_id}, session_data={self.session_data}, is_active={self.is_active})"

    @property
    def is_active(self):
        """Returns whether the session is active - made property to prevent tampering"""
        return self._is_active

    def create_message(self, is_user: bool, content: str):
        """Adds a message to the session

        Args:
            is_user (bool): Whether the message is from the user
            content (str): The content of the message

        Returns:
            Message: The Message object of the added message

        """
        if not self.is_active:
            raise Exception("Session is inactive")
        data = {"is_user": is_user, "content": content}
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages"
        response = self.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return Message(session_id=self.id, id=data["id"], is_user=is_user, content=content, created_at=data["created_at"])

    def get_message(self, message_id: uuid.UUID) -> Message:
        """Get a specific message for a session based on ID

        Args:
            message_id (uuid.UUID): The ID of the Message to retrieve

        Returns:
            Message: The Message object

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages/{message_id}"
        response = self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Message(session_id=self.id, id=data["id"], is_user=data["is_user"], content=data["content"], created_at=data["created_at"])

    def get_messages(self, page: int = 1, page_size: int = 50) -> GetMessagePage:
        """Get all messages for a session

        Args:
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return per page

        Returns:
            GetMessagePage: Page of Message objects

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages?page={page}&size={page_size}"
        response = self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return GetMessagePage(self, data)
        
    def get_messages_generator(self):
        """Shortcut Generator for get_messages. Generator to iterate through all messages for a session in an app

        Yields:
            Message: The Message object of the next Message

        """
        page = 1
        page_size = 50
        get_messages_page= self.get_messages(page, page_size)
        while True:
            # get_session_response = self.get_sessions(user_id, location_id, page, page_size)
            for message in get_messages_page.items:
                yield message

            new_messages = get_messages_page.next()
            if not new_messages:
                break
           
            get_messages_page = new_messages

    def create_metamessage(self, message: Message, metamessage_type: str, content: str):
        """Adds a metamessage to a session and links it to a specific message

        Args:
            message (Message): A message to associate the metamessage with
            metamessage_type (str): The type of the metamessage arbitrary itentifier
            content (str): The content of the metamessage

        Returns:
            Metamessage: The Metamessage object of the added metamessage

        """
        if not self.is_active:
            raise Exception("Session is inactive")
        data = {"metamessage_type": metamessage_type, "content": content, "message_id": message.id}
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/metamessages"
        response = self.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return Metamessage(id=data["id"], message_id=message.id, metamessage_type=metamessage_type, content=content, created_at=data["created_at"])


    def get_metamessage(self, metamessage_id: uuid.UUID) -> Metamessage:
        """Get a specific metamessage

        Args:
            message_id (uuid.UUID): The ID of the Message to retrieve

        Returns:
            Message: The Message object

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/metamessages/{metamessage_id}"
        response = self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Metamessage(id=data["id"], message_id=data["message_id"], metamessage_type=data["metamessage_type"], content=data["content"], created_at=data["created_at"])

    def get_metamessages(self, metamessage_type: Optional[str] = None, message: Optional[Message] = None, page: int = 1, page_size: int = 50) -> GetMetamessagePage:
        """Get all messages for a session

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            list[Dict]: List of Message objects

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/metamessages?page={page}&size={page_size}"
        if metamessage_type:
            url += f"&metamessage_type={metamessage_type}"
        if message:
            url += f"&message_id={message.id}"
        response = self.client.get(url)
        response.raise_for_status()
        data = response.json()
        options = {
                "metamessage_type": metamessage_type,
                "message_id": message.id if message else None
                }
        return GetMetamessagePage(self, options, data)
        
    def get_metamessages_generator(self, metamessage_type: Optional[str] = None, message: Optional[Message] = None):
        """Shortcut Generator for get_metamessages. Generator to iterate through all metamessages for a session in an app

        Args:
            metamessage_type (str, optional): Optional Metamessage type to filter by
            message (Message, optional): Optional Message to filter by

        Yields:
            Metamessage: The next Metamessage object of the requested query

        """
        page = 1
        page_size = 50
        get_metamessages_page = self.get_metamessages(metamessage_type=metamessage_type, message=message, page=page, page_size=page_size)
        while True:
            # get_session_response = self.get_sessions(user_id, location_id, page, page_size)
            for metamessage in get_metamessages_page.items:
                yield metamessage

            new_messages = get_metamessages_page.next()
            if not new_messages:
                break
           
            get_metamessages_page = new_messages

        
    def update(self, session_data: Dict):
        """Update the session_data of a session

        Args:
            session_data (Dict): The Session object containing any new session_data


        Returns:
            boolean: Whether the session was successfully updated
        """
        info = {"session_data": session_data}
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}"
        response = self.client.put(url, json=info)
        success = response.status_code < 400
        self.session_data = session_data
        return success

    def close(self):
        """Closes a session by marking it as inactive"""
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}"
        response = self.client.delete(url)
        response.raise_for_status()
        self._is_active = False

