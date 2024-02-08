import json
import uuid
from typing import Dict, Optional 
import httpx
from .schemas import Message, Metamessage

class AsyncGetPage:
    def __init__(self, response: Dict) -> None:
        self.total = response["total"]
        self.page = response["page"]
        self.page_size = response["size"]
        self.pages = response["pages"]
        self.items =[]

    async def next(self):
        pass

class AsyncGetSessionPage(AsyncGetPage):
    def __init__(self, client, response: Dict):
        super().__init__(response)
        self.client = client
        # self.total = response["total"]
        # self.page = response["page"]
        # self.page_size = response["size"]
        # self.pages = response["pages"]
        self.items = [
            AsyncSession(
                client=client,
                id=session["id"],
                user_id=session["user_id"],
                location_id=session["location_id"],
                is_active=session["is_active"],
                session_data=session["session_data"],
            )
            for session in response["items"]
        ]
       
    async def next(self):
        if self.page >= self.pages:
            return None
        user_id = self.items[0].user_id
        location_id = self.items[0].location_id
        return await self.client.get_sessions(user_id, location_id, self.page + 1, self.page_size)

class AsyncGetMessagePage(AsyncGetPage):
    def __init__(self, session, response: Dict):
        super().__init__(response)
        self.session = session
        # self.total = response["total"]
        # self.page = response["page"]
        # self.page_size = response["size"]
        # self.pages = response["pages"]
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

    async def next(self):
        if self.page >= self.pages:
            return None
        return await self.session.get_messages((self.page + 1), self.page_size)

class AsyncGetMetamessagePage(AsyncGetPage):
    def __init__(self, session, response: Dict) -> None:
        super().__init__(response)
        self.session = session
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

    async def next(self):
        if self.page >= self.pages:
            return None
        return await self.session.get_metamessages((self.page + 1), self.page_size)



class AsyncClient:
    def __init__(self, app_id: str, base_url: str = "https://demo.honcho.dev"):
        """Constructor for Client"""
        self.base_url = base_url  # Base URL for the instance of the Honcho API
        self.app_id = app_id # Representing ID of the client application
        self.client = httpx.AsyncClient()

    @property
    def common_prefix(self):
        return f"{self.base_url}/apps/{self.app_id}"

    async def get_session(self, user_id: str, session_id: uuid.UUID):
        """Get a specific session for a user by ID

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            Dict: The Session object of the requested Session

        """
        url = f"{self.common_prefix}/users/{user_id}/sessions/{session_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncSession(
            client=self,
            id=data["id"],
            user_id=data["user_id"],
            location_id=data["location_id"],
            is_active=data["is_active"],
            session_data=data["session_data"],
        )

    async def get_sessions(self, user_id: str, location_id: Optional[str] = None, page: int = 1, page_size: int = 50):
        """Return sessions associated with a user

        Args:
            user_id (str): The User ID representing the user, managed by the user
            location_id (str, optional): Optional Location ID representing the location of a session

        Returns:
            list[Dict]: List of Session objects

        """
        url = f"{self.common_prefix}/users/{user_id}/sessions?page={page}&size={page_size}" + (
            f"&location_id={location_id}" if location_id else ""
        )
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncGetSessionPage(self, data)

    async def get_sessions_generator(self, user_id: str, location_id: Optional[str] = None):
        page = 1
        page_size = 50
        get_session_response = await self.get_sessions(user_id, location_id, page, page_size)
        while True:
            # get_session_response = self.get_sessions(user_id, location_id, page, page_size)
            for session in get_session_response.items:
                yield session

            new_sessions = await get_session_response.next()
            if not new_sessions:
                break
           
            get_session_response = new_sessions

    async def create_session(
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
        response = await self.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return AsyncSession(
            self,
            id=data["id"],
            user_id=user_id,
            location_id=location_id,
            session_data=session_data,
            is_active=data["is_active"],
        )


class AsyncSession:
    def __init__(
        self,
        client: AsyncClient,
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
        return f"{self.base_url}/apps/{self.app_id}"

    def __str__(self):
        return f"Session(id={self.id}, app_id={self.app_id}, user_id={self.user_id}, location_id={self.location_id}, session_data={self.session_data}, is_active={self.is_active})"

    @property
    def is_active(self):
        return self._is_active

    async def create_message(self, is_user: bool, content: str):
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
        response = await self.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return Message(session_id=self.id, id=data["id"], is_user=is_user, content=content, created_at=data["created_at"])

    async def get_message(self, message_id: uuid.UUID) -> Message:
        """Get a specific message for a session based on ID

        Args:
            message_id (uuid.UUID): The ID of the Message to retrieve

        Returns:
            Message: The Message object

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages/{message_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Message(session_id=self.id, id=data["id"], is_user=data["is_user"], content=data["content"], created_at=data["created_at"])

    async def get_messages(self, page: int = 1, page_size: int = 50) -> AsyncGetMessagePage:
        """Get all messages for a session

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            list[Dict]: List of Message objects

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages?page={page}&size={page_size}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncGetMessagePage(self, data)
        
    async def get_messages_generator(self):
        page = 1
        page_size = 50
        get_messages_page= await self.get_messages(page, page_size)
        while True:
            # get_session_response = self.get_sessions(user_id, location_id, page, page_size)
            for message in get_messages_page.items:
                yield message

            new_messages = await get_messages_page.next()
            if not new_messages:
                break
           
            get_messages_page = new_messages

    async def get_metamessage(self, metamessage_id: uuid.UUID) -> Metamessage:
        """Get a specific message for a session based on ID

        Args:
            message_id (uuid.UUID): The ID of the Message to retrieve

        Returns:
            Message: The Message object

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/metamessages/{metamessage_id}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Metamessage(id=data["id"], message_id=data["message_id"], metamessage_type=data["metamessage_type"], content=data["content"], created_at=data["created_at"])

    async def get_metamessages(self, page: int = 1, page_size: int = 50) -> AsyncGetMetamessagePage:
        """Get all messages for a session

        Args:
            user_id (str): The User ID representing the user, managed by the user
            session_id (int): The ID of the Session to retrieve

        Returns:
            list[Dict]: List of Message objects

        """
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}/messages?page={page}&size={page_size}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncGetMetamessagePage(self, data)
        
    async def get_metamessages_generator(self):
        page = 1
        page_size = 50
        get_messages_response = await self.get_metamessages(page, page_size)
        while True:
            # get_session_response = self.get_sessions(user_id, location_id, page, page_size)
            for message in get_messages_response.items:
                yield message

            new_messages = await get_messages_response.next()
            if not new_messages:
                break
           
            get_messages_response = new_messages

        
    async def update(self, session_data: Dict):
        """Update the session_data of a session

        Args:
            session_data (Dict): The Session object containing any new session_data


        Returns:
            boolean: Whether the session was successfully updated
        """
        info = {"session_data": session_data}
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}"
        response = await self.client.put(url, json=info)
        success = response.status_code < 400
        self.session_data = session_data
        return success

    async def close(self):
        """Closes a session by marking it as inactive"""
        url = f"{self.common_prefix}/users/{self.user_id}/sessions/{self.id}"
        response = await self.client.delete(url)
        self._is_active = False

