from metacognition_sdk.interfaces import UserModelStorageAdapter


class InMemoryUserModelStorageAdapter(UserModelStorageAdapter):
    """
    Adapter for storing user models in memory.

    This adapter implements the UserModelStorageAdapter interface and provides
    a simple in-memory storage mechanism for user models.
    """

    user_model = (
        UserModelStorageAdapter.default_user_model
    )  # In-memory storage for the user model

    async def get_user_model(self) -> str:
        """
        Retrieve the user model from in-memory storage.

        Returns:
            str: The current user model stored in memory.
        """
        return self.user_model

    async def set_user_model(self, new_user_model: str) -> None:
        """
        Update the user model in in-memory storage.

        Args:
            new_user_model (str): The new user model to store in memory.
        """
        self.user_model = new_user_model
