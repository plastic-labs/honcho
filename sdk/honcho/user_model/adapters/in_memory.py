from ..interfaces import UserModelStorageAdapter


class InMemoryUserModelStorageAdapter(UserModelStorageAdapter):
    """
    Adapter for storing user models in memory.

    This adapter implements the UserModelStorageAdapter interface and provides
    a simple in-memory storage mechanism for user models.
    """

    user_models = {}

    def __init__(self, user_id: str, user_model: str = None):
        """
        Initialize the InMemoryUserModelStorageAdapter with a user ID and an optional initial user model.

        Args:
            user_id (str): The unique identifier for the user.
            user_model (str, optional): The initial model of the user. Defaults to the class's default user model.
        """
        self.user_id = user_id
        self.user_model = user_model or self.user_models.get(
            user_id, UserModelStorageAdapter.default_user_model
        )

    async def get_user_model(self) -> str:
        """
        Retrieve the user model from in-memory storage.

        Returns:
            str: The current user model stored in memory.
        """
        return self._user_models.get(self.user_id, self.default_user_model)

    async def set_user_model(self, new_user_model: str) -> None:
        """
        Update the user model in in-memory storage.

        Args:
            new_user_model (str): The new user model to store in memory.
        """
        self._user_models[self.user_id] = new_user_model
