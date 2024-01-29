from abc import ABC, abstractmethod


class UserModelStorageAdapter(ABC):
    """
    Abstract base class that defines the interface for user model storage adapters that define how and where a user model is stored
    """

    default_user_model = "none, new user"

    @abstractmethod
    def __init__(self, user_id: str, user_model: str = None):
        """
        Initialize the storage adapter with a user ID and an initial user model.

        Args:
            user_id (str): The unique identifier for the user.
            user_model (str): The initial model of the user.
        """

        pass

    @abstractmethod
    async def get_user_model(self) -> str:
        """
        Retrieve the user model from storage.

        Returns:
            str: The user model as a string.
        """
        pass

    @abstractmethod
    async def set_user_model(self, new_user_model: str) -> None:
        """
        Update the user model in storage.

        Args:
            new_user_model (str): The new user model as a string.
        """
        pass
