import os
import pickle
from ..interfaces import UserModelStorageAdapter


class PickleUserModelStorageAdapter(UserModelStorageAdapter):
    """
    Adapter for storing user models using Python's pickle module.
    """

    def __init__(self, user_id: str):
        """
        Initialize the PickleUserModelStorageAdapter with a user ID.

        Args:
            user_id (str): The unique identifier for the user.
        """
        self.user_id = user_id
        self.filepath = f"user_model_{user_id}.pkl"
        if not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                pickle.dump({}, f)

    async def get_user_model(self) -> str:
        """
        Retrieve the user model from the pickle file.

        Returns:
            str: The current user model for the given user ID.
        """
        with open(self.filepath, "rb") as f:
            user_models = pickle.load(f)
        return user_models.get(self.user_id, UserModelStorageAdapter.default_user_model)

    async def set_user_model(self, new_user_model: str) -> None:
        """
        Update the user model in the pickle file.

        Args:
            new_user_model (str): The new user model to store.
        """
        with open(self.filepath, "rb") as f:
            user_models = pickle.load(f)
        user_models[self.user_id] = new_user_model
        with open(self.filepath, "wb") as f:
            pickle.dump(user_models, f)
