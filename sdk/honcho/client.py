from __future__ import annotations

import datetime
import json
import uuid
from typing import Optional

import httpx

from .schemas import Document, Message, Metamessage


class AsyncGetPage:
    """Base class for receiving Paginated API results"""

    def __init__(self, response: dict) -> None:
        """Constructor for Page with relevant information about the results and pages

        Args:
            response (Dict): Response from API with pagination information
        """
        self.total = response["total"]
        self.page = response["page"]
        self.page_size = response["size"]
        self.pages = response["pages"]
        self.items = []

    async def next(self):
        """Shortcut method to Get the next page of results"""
        pass


class AsyncGetUserPage(AsyncGetPage):
    """Paginated Results for Get User Requests"""

    def __init__(
        self,
        response: dict,
        honcho: AsyncHoncho,
        filter: Optional[dict],
        reverse: bool,
    ):
        """Constructor for Page Result from User Get Request

        Args:
            response (dict): Response from API with pagination information
            honcho (AsyncHoncho): Honcho Client
            reverse (bool): Whether to reverse the order of the results or not
        """
        super().__init__(response)
        self.honcho = honcho
        self.filter = filter
        self.reverse = reverse
        self.items = [
            AsyncUser(
                honcho=honcho,
                id=user["id"],
                created_at=user["created_at"],
                metadata=user["metadata"],
            )
            for user in response["items"]
        ]

    async def next(self):
        if self.page >= self.pages:
            return None
        return await self.honcho.get_users(
            filter=self.filter,
            page=(self.page + 1),
            page_size=self.page_size,
            reverse=self.reverse,
        )


class AsyncGetSessionPage(AsyncGetPage):
    """Paginated Results for Get Session Requests"""

    def __init__(
        self,
        response: dict,
        user: AsyncUser,
        reverse: bool,
        location_id: Optional[str],
        filter: Optional[dict],
        is_active: bool,
    ):
        """Constructor for Page Result from Session Get Request

        Args:
            response (dict): Response from API with pagination information
            user (AsyncUser): Honcho User associated with the session
            reverse (bool): Whether to reverse the order of the results or not
            location_id (str): ID of the location associated with the session
        """
        super().__init__(response)
        self.user = user
        self.location_id = location_id
        self.reverse = reverse
        self.is_active = is_active
        self.filter = filter
        self.items = [
            AsyncSession(
                user=user,
                id=session["id"],
                location_id=session["location_id"],
                is_active=session["is_active"],
                metadata=session["metadata"],
                created_at=session["created_at"],
            )
            for session in response["items"]
        ]

    async def next(self):
        """Get the next page of results

        Returns:
            AsyncGetSessionPage | None: Next Page of Results or None if there are no more sessions to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return await self.user.get_sessions(
            location_id=self.location_id,
            filter=self.filter,
            page=(self.page + 1),
            page_size=self.page_size,
            reverse=self.reverse,
            is_active=self.is_active,
        )


class AsyncGetMessagePage(AsyncGetPage):
    """Paginated Results for Get Session Requests"""

    def __init__(
        self,
        response: dict,
        session: AsyncSession,
        filter: Optional[dict],
        reverse: bool,
    ):
        """Constructor for Page Result from Session Get Request

        Args:
            response (dict): Response from API with pagination information
            session (AsyncSession): Session the returned messages are associated with
            reverse (bool): Whether to reverse the order of the results or not
        """
        super().__init__(response)
        self.session = session
        self.filter = filter
        self.reverse = reverse
        self.items = [
            Message(
                session_id=session.id,
                id=message["id"],
                is_user=message["is_user"],
                content=message["content"],
                metadata=message["metadata"],
                created_at=message["created_at"],
            )
            for message in response["items"]
        ]

    async def next(self):
        """Get the next page of results

        Returns:
            AsyncGetMessagePage | None: Next Page of Results or None if there
            are no more messages to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return await self.session.get_messages(
            self.filter, (self.page + 1), self.page_size, self.reverse
        )


class AsyncGetMetamessagePage(AsyncGetPage):
    def __init__(
        self,
        response: dict,
        session,
        filter: Optional[dict],
        reverse: bool,
        message_id: Optional[uuid.UUID],
        metamessage_type: Optional[str],
    ) -> None:
        """Constructor for Page Result from Metamessage Get Request

        Args:
            response (dict): Response from API with pagination information
            session (AsyncSession): Session the returned messages are
            associated with
            reverse (bool): Whether to reverse the order of the results
            message_id (Optional[str]): ID of the message associated with the
            metamessage_type (Optional[str]): Type of the metamessage
        """
        super().__init__(response)
        self.session = session
        self.message_id = message_id
        self.metamessage_type = metamessage_type
        self.filter = filter
        self.reverse = reverse
        self.items = [
            Metamessage(
                id=metamessage["id"],
                message_id=metamessage["message_id"],
                metadata=metamessage["metadata"],
                metamessage_type=metamessage["metamessage_type"],
                content=metamessage["content"],
                created_at=metamessage["created_at"],
            )
            for metamessage in response["items"]
        ]

    async def next(self):
        """Get the next page of results

        Returns:
            AsyncGetMetamessagePage | None: Next Page of Results or None if
            there are no more metamessages to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return await self.session.get_metamessages(
            metamessage_type=self.metamessage_type,
            filter=self.filter,
            message=self.message_id,
            page=(self.page + 1),
            page_size=self.page_size,
            reverse=self.reverse,
        )


class AsyncGetDocumentPage(AsyncGetPage):
    """Paginated results for Get Document requests"""

    def __init__(
        self, response: dict, collection, filter: Optional[dict], reverse: bool
    ) -> None:
        """Constructor for Page Result from Document Get Request

        Args:
            response (dict): Response from API with pagination information
            collection (AsyncCollection): Collection the returned documents are
            associated with
            reverse (bool): Whether to reverse the order of the results or not
        """
        super().__init__(response)
        self.collection = collection
        self.filter = filter
        self.reverse = reverse
        self.items = [
            Document(
                id=document["id"],
                collection_id=collection.id,
                content=document["content"],
                metadata=document["metadata"],
                created_at=document["created_at"],
            )
            for document in response["items"]
        ]

    async def next(self):
        """Get the next page of results

        Returns:
            AsyncGetDocumentPage | None: Next Page of Results or None if there
            are no more sessions to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return await self.collection.get_documents(
            filter=self.filter,
            page=self.page + 1,
            page_size=self.page_size,
            reverse=self.reverse,
        )


class AsyncGetCollectionPage(AsyncGetPage):
    """Paginated results for Get Collection requests"""

    def __init__(
        self, response: dict, user: AsyncUser, filter: Optional[dict], reverse: bool
    ):
        """Constructor for page result from Get Collection Request

        Args:
            response (dict): Response from API with pagination information
            user (AsyncUser): Honcho Client
            reverse (bool): Whether to reverse the order of the results or not
        """
        super().__init__(response)
        self.user = user
        self.filter = filter
        self.reverse = reverse
        self.items = [
            AsyncCollection(
                user=user,
                id=collection["id"],
                name=collection["name"],
                metadata=collection["metadata"],
                created_at=collection["created_at"],
            )
            for collection in response["items"]
        ]

    async def next(self):
        """Get the next page of results

        Returns:
            AsyncGetCollectionPage | None: Next Page of Results or None if
            there are no more sessions to retreive from a query
        """
        if self.page >= self.pages:
            return None
        return await self.user.get_collections(
            filter=self.filter,
            page=self.page + 1,
            page_size=self.page_size,
            reverse=self.reverse,
        )


class AsyncHoncho:
    """Honcho API Client Object"""

    def __init__(self, app_name: str, base_url: str = "https://demo.honcho.dev"):
        """Constructor for Client"""
        self.server_url: str = base_url  # Base URL for the instance of the Honcho API
        self.client: httpx.AsyncClient = httpx.AsyncClient()
        self.app_name: str = app_name  # Representing name of the client application
        self.app_id: uuid.UUID
        self.metadata: dict

    async def initialize(self):
        res = await self.client.get(
            f"{self.server_url}/apps/get_or_create/{self.app_name}"
        )
        res.raise_for_status()
        data = res.json()
        self.app_id: uuid.UUID = data["id"]
        self.metadata: dict = data["metadata"]

    @property
    def base_url(self):
        """Shorcut for common API prefix. made a property to prevent tampering"""
        return f"{self.server_url}/apps/{self.app_id}"

    async def update(self, metadata: dict):
        """Update the metadata of the app associated with this instance of the Honcho
        client

        Args:
            metadata (dict): The metadata to update

        Returns:
            boolean: Whether the metadata was successfully updated
        """
        data = {"metadata": metadata}
        url = f"{self.base_url}"
        response = await self.client.put(url, json=data)
        success = response.status_code < 400
        self.metadata = metadata
        return success

    async def create_user(self, name: str, metadata: Optional[dict] = None):
        """Create a new user by name

        Args:
            name (str): The name of the user
            metadata (dict, optional): The metadata for the user. Defaults to {}.

        Returns:
            AsyncUser: The created User object
        """
        if metadata is None:
            metadata = {}
        url = f"{self.base_url}/users"
        response = await self.client.post(
            url, json={"name": name, "metadata": metadata}
        )
        response.raise_for_status()
        data = response.json()
        return AsyncUser(
            honcho=self,
            id=data["id"],
            metadata=data["metadata"],
            created_at=data["created_at"],
        )

    async def get_user(self, name: str):
        """Get a user by name

        Args:
            name (str): The name of the user

        Returns:
            AsyncUser: The User object
        """
        url = f"{self.base_url}/users/{name}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncUser(
            honcho=self,
            id=data["id"],
            metadata=data["metadata"],
            created_at=data["created_at"],
        )

    async def get_or_create_user(self, name: str):
        """Get or Create a user by name

        Args:
            name (str): The name of the user

        Returns:
            AsyncUser: The User object
        """
        url = f"{self.base_url}/users/get_or_create/{name}"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncUser(
            honcho=self,
            id=data["id"],
            metadata=data["metadata"],
            created_at=data["created_at"],
        )

    async def get_users(
        self,
        filter: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
        reverse: bool = False,
    ):
        """Get Paginated list of users

        Args:
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return
            reverse (bool): Whether to reverse the order of the results

        Returns:
            AsyncGetUserPage: Paginated list of users
        """
        # url = f"{self.base_url}/users?page={page}&size={page_size}&reverse={reverse}"
        url = f"{self.base_url}/users"
        params = {
            "page": page,
            "size": page_size,
            "reverse": reverse,
        }
        if filter is not None:
            json_filter = json.dumps(filter)
            params["filter"] = json_filter
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return AsyncGetUserPage(data, self, filter, reverse)

    async def get_users_generator(
        self,
        filter: Optional[dict] = None,
        reverse: bool = False,
    ):
        """Shortcut Generator for get_users. Generator to iterate through
        all users in an app

        Args:
            reverse (bool): Whether to reverse the order of the results

        Yields:
            AsyncUser: The User object

        """
        page = 1
        page_size = 50
        get_user_response = await self.get_users(filter, page, page_size, reverse)
        while True:
            for session in get_user_response.items:
                yield session

            new_users = await get_user_response.next()
            if not new_users:
                break

            get_user_response = new_users

    # async def get_user_by_id(self, id: uuid.UUID):
    #     """Get a user by id

    #     Args:
    #         id (uuid.UUID): The id of the user

    #     Returns:
    #         AsyncUser: The User object
    #     """
    #     url = f"{self.common_prefix}/users/{id}"
    #     response = await self.client.get(url)
    #     response.raise_for_status()
    #     data = response.json()
    #     return AsyncUser(self, **data)


class AsyncUser:
    """Represents a single user in an app"""

    def __init__(
        self,
        honcho: AsyncHoncho,
        id: uuid.UUID,
        metadata: dict,
        created_at: datetime.datetime,
    ):
        """Constructor for User"""
        # self.base_url: str = honcho.base_url
        self.honcho: AsyncHoncho = honcho
        self.id: uuid.UUID = id
        self.metadata: dict = metadata
        self.created_at: datetime.datetime = created_at

    @property
    def base_url(self):
        """Shortcut for common API prefix. made a property to prevent tampering"""
        return f"{self.honcho.base_url}/users/{self.id}"

    def __str__(self):
        """String representation of User"""
        return f"AsyncUser(id={self.id}, app_id={self.honcho.app_id}, metadata={self.metadata})"  # noqa: E501

    async def update(self, metadata: dict):
        """Updates a user's metadata

        Args:
            metadata (dict): The new metadata for the user

        Returns:
            AsyncUser: The updated User object

        """
        data = {"metadata": metadata}
        url = f"{self.base_url}"
        response = await self.honcho.client.put(url, json=data)
        response.raise_for_status()
        success = response.status_code < 400
        data = response.json()
        self.metadata = data["metadata"]
        return success
        # return AsyncUser(self.honcho, **data)

    async def get_session(self, session_id: uuid.UUID):
        """Get a specific session for a user by ID

        Args:
            session_id (uuid.UUID): The ID of the Session to retrieve

        Returns:
            AsyncSession: The Session object of the requested Session

        """
        url = f"{self.base_url}/sessions/{session_id}"
        response = await self.honcho.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncSession(
            user=self,
            id=data["id"],
            location_id=data["location_id"],
            is_active=data["is_active"],
            metadata=data["metadata"],
            created_at=data["created_at"],
        )

    async def get_sessions(
        self,
        location_id: Optional[str] = None,
        filter: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
        reverse: bool = False,
        is_active: bool = False,
    ):
        """Return sessions associated with a user paginated

        Args:
            location_id (str, optional): Optional Location ID representing the
            location of a session
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return
            reverse (bool): Whether to reverse the order of the results
            is_active (bool): Whether to only return active sessions

        Returns:
            AsyncGetSessionPage: Page or results for get_sessions query

        """
        # url = (
        #     f"{self.base_url}/sessions?page={page}&size={page_size}&reverse={reverse}&is_active={is_active}"
        #     + (f"&location_id={location_id}" if location_id else "")
        # )
        url = f"{self.base_url}/sessions"
        params = {
            "page": page,
            "size": page_size,
            "reverse": reverse,
            "is_active": is_active,
        }
        if location_id:
            params["location_id"] = location_id
        if filter is not None:
            json_filter = json.dumps(filter)
            params["filter"] = json_filter
        response = await self.honcho.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return AsyncGetSessionPage(data, self, reverse, location_id, filter, is_active)

    async def get_sessions_generator(
        self,
        location_id: Optional[str] = None,
        reverse: bool = False,
        is_active: bool = False,
        filter: Optional[dict] = None,
    ):
        """Shortcut Generator for get_sessions. Generator to iterate through
        all sessions for a user in an app

        Args:
            location_id (str, optional): Optional Location ID representing the
            location of a session
            reverse (bool): Whether to reverse the order of the results
            is_active (bool): Whether to only return active sessions

        Yields:
            AsyncSession: The Session object of the requested Session

        """
        page = 1
        page_size = 50
        get_session_response = await self.get_sessions(
            location_id, filter, page, page_size, reverse, is_active
        )
        while True:
            for session in get_session_response.items:
                yield session

            new_sessions = await get_session_response.next()
            if not new_sessions:
                break

            get_session_response = new_sessions

    async def create_session(
        self, location_id: str = "default", metadata: Optional[dict] = None
    ):
        """Create a session for a user

        Args:
            location_id (str, optional): Optional Location ID representing the
            location of a session
            metadata (dict, optional): Optional session metadata

        Returns:
            AsyncSession: The Session object of the new Session

        """
        if metadata is None:
            metadata = {}
        data = {"location_id": location_id, "metadata": metadata}
        url = f"{self.base_url}/sessions"
        response = await self.honcho.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return AsyncSession(
            self,
            id=data["id"],
            location_id=location_id,
            metadata=metadata,
            is_active=data["is_active"],
            created_at=data["created_at"],
        )

    async def create_collection(
        self,
        name: str,
        metadata: Optional[dict] = None,
    ):
        """Create a collection for a user

        Args:
            name (str): unique name for the collection for the user

        Returns:
            AsyncCollection: The Collection object of the new Collection

        """
        if metadata is None:
            metadata = {}
        data = {"name": name, "metadata": metadata}
        url = f"{self.base_url}/collections"
        response = await self.honcho.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return AsyncCollection(
            self,
            id=data["id"],
            name=name,
            metadata=metadata,
            created_at=data["created_at"],
        )

    async def get_collection(self, name: str):
        """Get a specific collection for a user by name

        Args:
            name (str): The name of the collection to get

        Returns:
            AsyncCollection: The Session object of the requested Session

        """
        url = f"{self.base_url}/collections/{name}"
        response = await self.honcho.client.get(url)
        response.raise_for_status()
        data = response.json()
        return AsyncCollection(
            user=self,
            id=data["id"],
            name=data["name"],
            metadata=data["metadata"],
            created_at=data["created_at"],
        )

    async def get_collections(
        self,
        filter: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
        reverse: bool = False,
    ):
        """Return collections associated with a user paginated

        Args:
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return
            reverse (bool): Whether to reverse the order of the results

        Returns:
            AsyncGetCollectionPage: Page or results for get_collections query

        """
        # url = f"{self.base_url}/collections?page={page}&size={page_size}&reverse={reverse}"  # noqa: E501
        url = f"{self.base_url}/collections"
        params = {
            "page": page,
            "size": page_size,
            "reverse": reverse,
        }
        if filter is not None:
            json_filter = json.dumps(filter)
            params["filter"] = json_filter
        response = await self.honcho.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return AsyncGetCollectionPage(data, self, filter, reverse)

    async def get_collections_generator(
        self, filter: Optional[dict] = None, reverse: bool = False
    ):
        """Shortcut Generator for get_sessions. Generator to iterate through
        all sessions for a user in an app

        Args:
            reverse (bool): Whether to reverse the order of the results

        Yields:
            AsyncCollection: The Session object of the requested Session

        """
        page = 1
        page_size = 50
        get_collection_response = await self.get_collections(
            filter, page, page_size, reverse
        )
        while True:
            for collection in get_collection_response.items:
                yield collection

            new_collections = await get_collection_response.next()
            if not new_collections:
                break

            get_collection_response = new_collections


class AsyncSession:
    """Represents a single session for a user in an app"""

    def __init__(
        self,
        user: AsyncUser,
        id: uuid.UUID,
        location_id: str,
        metadata: dict,
        is_active: bool,
        created_at: datetime.datetime,
    ):
        """Constructor for Session"""
        self.user: AsyncUser = user
        self.id: uuid.UUID = id
        self.location_id: str = location_id
        self.metadata: dict = metadata
        self._is_active: bool = is_active
        self.created_at: datetime.datetime = created_at

    @property
    def base_url(self):
        """Shortcut for common API prefix. made a property to prevent tampering"""
        return f"{self.user.base_url}/sessions/{self.id}"

    def __str__(self):
        """String representation of Session"""
        return f"AsyncSession(id={self.id}, location_id={self.location_id}, metadata={self.metadata}, is_active={self.is_active})"  # noqa: E501

    @property
    def is_active(self):
        """Returns whether the session is active - made property to prevent tampering"""
        return self._is_active

    async def create_message(
        self, is_user: bool, content: str, metadata: Optional[dict] = None
    ):
        """Adds a message to the session

        Args:
            is_user (bool): Whether the message is from the user
            content (str): The content of the message

        Returns:
            Message: The Message object of the added message

        """
        if not self.is_active:
            raise Exception("Session is inactive")
        if metadata is None:
            metadata = {}
        data = {"is_user": is_user, "content": content, "metadata": metadata}
        url = f"{self.base_url}/messages"
        response = await self.user.honcho.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return Message(
            session_id=self.id,
            id=data["id"],
            is_user=is_user,
            content=content,
            metadata=metadata,
            created_at=data["created_at"],
        )

    async def get_message(self, message_id: uuid.UUID) -> Message:
        """Get a specific message for a session based on ID

        Args:
            message_id (uuid.UUID): The ID of the Message to retrieve

        Returns:
            Message: The Message object

        """
        url = f"{self.base_url}/messages/{message_id}"
        response = await self.user.honcho.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Message(
            session_id=self.id,
            id=data["id"],
            is_user=data["is_user"],
            content=data["content"],
            metadata=data["metadata"],
            created_at=data["created_at"],
        )

    async def get_messages(
        self,
        filter: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
        reverse: bool = False,
    ) -> AsyncGetMessagePage:
        """Get all messages for a session

        Args:
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return per page
            reverse (bool): Whether to reverse the order of the results

        Returns:
            AsyncGetMessagePage: Page of Message objects

        """
        url = f"{self.base_url}/messages"
        params = {
            "page": page,
            "size": page_size,
            "reverse": reverse,
        }
        if filter is not None:
            json_filter = json.dumps(filter)
            params["filter"] = json_filter
        response = await self.user.honcho.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return AsyncGetMessagePage(data, self, filter, reverse)

    async def get_messages_generator(
        self, filter: Optional[dict] = None, reverse: bool = False
    ):
        """Shortcut Generator for get_messages. Generator to iterate through
        all messages for a session in an app

        Yields:
            Message: The Message object of the next Message

        """
        page = 1
        page_size = 50
        get_messages_page = await self.get_messages(filter, page, page_size, reverse)
        while True:
            for message in get_messages_page.items:
                yield message

            new_messages = await get_messages_page.next()
            if not new_messages:
                break

            get_messages_page = new_messages

    async def create_metamessage(
        self,
        message: Message,
        metamessage_type: str,
        content: str,
        metadata: Optional[dict] = None,
    ):
        """Adds a metamessage to a session and links it to a specific message

        Args:
            message (Message): A message to associate the metamessage with
            metamessage_type (str): The type of the metamessage arbitrary identifier
            content (str): The content of the metamessage

        Returns:
            Metamessage: The Metamessage object of the added metamessage

        """
        if not self.is_active:
            raise Exception("Session is inactive")
        if metadata is None:
            metadata = {}
        data = {
            "metamessage_type": metamessage_type,
            "content": content,
            "message_id": message.id,
            "metadata": metadata,
        }
        url = f"{self.base_url}/metamessages"
        response = await self.user.honcho.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return Metamessage(
            id=data["id"],
            message_id=message.id,
            metamessage_type=metamessage_type,
            content=content,
            metadata=metadata,
            created_at=data["created_at"],
        )

    async def get_metamessage(self, metamessage_id: uuid.UUID) -> Metamessage:
        """Get a specific metamessage

        Args:
            message_id (uuid.UUID): The ID of the Message to retrieve

        Returns:
            Message: The Message object

        """
        url = f"{self.base_url}/metamessages/{metamessage_id}"
        response = await self.user.honcho.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Metamessage(
            id=data["id"],
            message_id=data["message_id"],
            metamessage_type=data["metamessage_type"],
            content=data["content"],
            metadata=data["metadata"],
            created_at=data["created_at"],
        )

    async def get_metamessages(
        self,
        metamessage_type: Optional[str] = None,
        message: Optional[Message] = None,
        filter: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
        reverse: bool = False,
    ) -> AsyncGetMetamessagePage:
        """Get all messages for a session

        Args:
            metamessage_type (str, optional): The type of the metamessage
            message (Message, optional): The message to associate the metamessage with
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return per page
            reverse (bool): Whether to reverse the order of the results

        Returns:
            list[dict]: List of Message objects

        """
        # url = f"{self.base_url}/metamessages?page={page}&size={page_size}&reverse={reverse}"  # noqa: E501
        url = f"{self.base_url}/metamessages"
        params = {
            "page": page,
            "size": page_size,
            "reverse": reverse,
        }
        if metamessage_type:
            # url += f"&metamessage_type={metamessage_type}"
            params["metamessage_type"] = metamessage_type
        if message:
            # url += f"&message_id={message.id}"
            params["message_id"] = message.id
        if filter is not None:
            json_metadata = json.dumps(filter)
            params["filter"] = json_metadata
        response = await self.user.honcho.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        message_id = message.id if message else None
        return AsyncGetMetamessagePage(
            data, self, filter, reverse, message_id, metamessage_type
        )

    async def get_metamessages_generator(
        self,
        metamessage_type: Optional[str] = None,
        message: Optional[Message] = None,
        filter: Optional[dict] = None,
        reverse: bool = False,
    ):
        """Shortcut Generator for get_metamessages. Generator to iterate
        through all metamessages for a session in an app

        Args:
            metamessage_type (str, optional): Optional Metamessage type to filter by
            message (Message, optional): Optional Message to filter by

        Yields:
            Metamessage: The next Metamessage object of the requested query

        """
        page = 1
        page_size = 50
        get_metamessages_page = await self.get_metamessages(
            metamessage_type=metamessage_type,
            message=message,
            filter=filter,
            page=page,
            page_size=page_size,
            reverse=reverse,
        )
        while True:
            for metamessage in get_metamessages_page.items:
                yield metamessage

            new_messages = await get_metamessages_page.next()
            if not new_messages:
                break

            get_metamessages_page = new_messages

    async def update(self, metadata: dict):
        """Update the metadata of a session

        Args:
            metadata (dict): The Session object containing any new metadata

        Returns:
            boolean: Whether the session was successfully updated
        """
        info = {"metadata": metadata}
        url = f"{self.base_url}"
        response = await self.user.honcho.client.put(url, json=info)
        success = response.status_code < 400
        self.metadata = metadata
        return success

    async def update_message(self, message: Message, metadata: dict):
        """Update the metadata of a message

        Args:
            message (Message): The message to update
            metadata (dict): The new metadata for the message

        Returns:
            boolean: Whether the message was successfully updated
        """
        info = {"metadata": metadata}
        url = f"{self.base_url}/messages/{message.id}"
        response = await self.user.honcho.client.put(url, json=info)
        success = response.status_code < 400
        message.metadata = metadata
        return success

    async def update_metamessage(
        self,
        metamessage: Metamessage,
        metamessage_type: Optional[str],
        metadata: Optional[dict],
    ):
        """Update the metadata of a metamessage

        Args:
            metamessage (Metamessage): The metamessage to update
            metadata (dict): The new metadata for the metamessage

        Returns:
            boolean: Whether the metamessage was successfully updated
        """
        if metadata is None and metamessage_type is None:
            raise ValueError("metadata and metamessage_type cannot both be None")
        info = {"metamessage_type": metamessage_type, "metadata": metadata}
        url = f"{self.base_url}/metamessages/{metamessage.id}"
        response = await self.user.honcho.client.put(url, json=info)
        success = response.status_code < 400
        if metamessage_type is not None:
            metamessage.metamessage_type = metamessage_type
        if metadata is not None:
            metamessage.metadata = metadata
        return success

    async def close(self):
        """Closes a session by marking it as inactive"""
        url = f"{self.base_url}"
        response = await self.user.honcho.client.delete(url)
        response.raise_for_status()
        self._is_active = False

    async def chat(self, query) -> str:
        url = f"{self.base_url}/chat"
        params = {"query": query}
        response = await self.user.honcho.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data["content"]


class AsyncCollection:
    """Represents a single collection for a user in an app"""

    def __init__(
        self,
        user: AsyncUser,
        id: uuid.UUID,
        name: str,
        metadata: dict,
        created_at: datetime.datetime,
    ):
        """Constructor for Collection"""
        self.user = user
        self.id: uuid.UUID = id
        self.name: str = name
        self.metadata: dict = metadata
        self.created_at: datetime.datetime = created_at

    @property
    def base_url(self):
        """Shortcut for common API prefix. made a property to prevent tampering"""
        return f"{self.user.base_url}/collections/{self.id}"

    def __str__(self):
        """String representation of Collection"""
        return f"AsyncCollection(id={self.id}, name={self.name}, created_at={self.created_at})"  # noqa: E501

    async def update(self, name: Optional[str] = None, metadata: Optional[dict] = None):
        """Update the name of the collection

        Args:
            name (str): The new name of the document

        Returns:
            boolean: Whether the session was successfully updated
        """
        if metadata is None and name is None:
            raise ValueError("metadata and name cannot both be None")
        info = {"name": name, "metadata": metadata}
        url = f"{self.base_url}"
        response = await self.user.honcho.client.put(url, json=info)
        response.raise_for_status()
        success = response.status_code < 400
        if name is not None:
            self.name = name
        if metadata is not None:
            self.metadata = metadata
        return success

    async def delete(self):
        """Delete a collection and all associated documents"""
        url = f"{self.base_url}"
        response = await self.user.honcho.client.delete(url)
        response.raise_for_status()

    async def create_document(self, content: str, metadata: Optional[dict] = None):
        """Adds a document to the collection

        Args:
            content (str): The content of the document
            metadata (dict): The metadata of the document

        Returns:
            Document: The Document object of the added document

        """
        if metadata is None:
            metadata = {}
        data = {"metadata": metadata, "content": content}
        url = f"{self.base_url}/documents"
        response = await self.user.honcho.client.post(url, json=data)
        response.raise_for_status()
        data = response.json()
        return Document(
            collection_id=self.id,
            id=data["id"],
            metadata=metadata,
            content=content,
            created_at=data["created_at"],
        )

    async def get_document(self, document_id: uuid.UUID) -> Document:
        """Get a specific document for a collection based on ID

        Args:
            document_id (uuid.UUID): The ID of the Document to retrieve

        Returns:
            Document: The Document object

        """
        url = f"{self.base_url}/documents/{document_id}"
        response = await self.user.honcho.client.get(url)
        response.raise_for_status()
        data = response.json()
        return Document(
            collection_id=self.id,
            id=data["id"],
            metadata=data["metadata"],
            content=data["content"],
            created_at=data["created_at"],
        )

    async def get_documents(
        self,
        filter: Optional[dict] = None,
        page: int = 1,
        page_size: int = 50,
        reverse: bool = False,
    ) -> AsyncGetDocumentPage:
        """Get all documents for a collection

        Args:
            page (int, optional): The page of results to return
            page_size (int, optional): The number of results to return per page

        Returns:
            AsyncGetDocumentPage: Page of Document objects

        """
        # url = f"{self.base_url}/documents?page={page}&size={page_size}&reverse={reverse}"  # noqa: E501
        url = f"{self.base_url}/documents"
        params = {
            "page": page,
            "size": page_size,
            "reverse": reverse,
        }
        if filter is not None:
            json_filter = json.dumps(filter)
            params["filter"] = json_filter
        response = await self.user.honcho.client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return AsyncGetDocumentPage(data, self, filter, reverse)

    async def get_documents_generator(
        self, filter: Optional[dict] = None, reverse: bool = False
    ):
        """Shortcut Generator for get_documents. Generator to iterate through
        all documents for a collection in an app

        Yields:
            Document: The Document object of the next Document

        """
        page = 1
        page_size = 50
        get_documents_page = await self.get_documents(filter, page, page_size, reverse)
        while True:
            for document in get_documents_page.items:
                yield document

            new_documents = await get_documents_page.next()
            if not new_documents:
                break

            get_documents_page = new_documents

    async def query(self, query: str, top_k: int = 5) -> list[Document]:
        """query the documents by cosine distance
        Args:
            query (str): The query string to compare other embeddings too
            top_k (int, optional): The number of results to return. Defaults to 5 max 50

        Returns:
            List[Document]: The response from the query with matching documents
        """
        # url = f"{self.base_url}/query?query={query}&top_k={top_k}"
        url = f"{self.base_url}/query"
        params = {"query": query, "top_k": top_k}
        response = await self.user.honcho.client.get(url, params=params)
        response.raise_for_status()
        data = [
            Document(
                collection_id=self.id,
                content=document["content"],
                id=document["id"],
                created_at=document["created_at"],
                metadata=document["metadata"],
            )
            for document in response.json()
        ]
        return data

    async def update_document(
        self,
        document: Document,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Document:
        """Update a document in the collection

        Args:
            document (Document): The Document to update
            metadata (dict): The metadata of the document
            content (str): The content of the document

        Returns:
            Document: The newly updated Document
        """
        if metadata is None and content is None:
            raise ValueError("metadata and content cannot both be None")
        data = {"metadata": metadata, "content": content}
        url = f"{self.base_url}/documents/{document.id}"
        response = await self.user.honcho.client.put(url, json=data)
        response.raise_for_status()
        data = response.json()
        return Document(
            data["id"],
            metadata=data["metadata"],
            content=data["content"],
            created_at=data["created_at"],
            collection_id=data["collection_id"],
        )

    async def delete_document(self, document: Document) -> bool:
        """Delete a document from the collection

        Args:
            document (Document): The Document to delete

        Returns:
            boolean: Whether the document was successfully deleted
        """
        url = f"{self.base_url}/documents/{document.id}"
        response = await self.user.honcho.client.delete(url)
        response.raise_for_status()
        success = response.status_code < 400
        return success
