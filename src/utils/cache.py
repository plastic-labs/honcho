from collections import OrderedDict
import asyncio

class LRUCache:
    """
    A simple asyncio-safe LRU cache implementation.
    
    This cache automatically evicts the least recently used items when it reaches
    its capacity limit. All operations are protected by an asyncio lock for
    concurrent access safety.
    """
    
    def __init__(self, capacity=100):
        """
        Initialize the LRU cache.
        
        Args:
            capacity (int): Maximum number of items to store in the cache
        """
        self._cache = OrderedDict()
        self._capacity = max(1, capacity)
        self._lock = asyncio.Lock()
        
    
    async def get(self, key):
        """
        Get a value from the cache.
        
        Args:
            key: The key to look up
            
        Returns:
            The cached value or None if not found
        """
        async with self._lock:
            if key not in self._cache:
                return None
            
            # Move accessed item to the end to mark as most recently used
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
    
    async def put(self, key, value):
        """
        Add or update an entry in the cache.
        
        Args:
            key: The key to store
            value: The value to store

        Returns:
            The value that was stored
        """
        async with self._lock:
            # If key already exists, remove it first to update its position
            if key in self._cache:
                self._cache.pop(key)
            
            # If at capacity, remove the least recently used item
            elif len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)
            
            # Add the new item
            self._cache[key] = value
            return value
    
    async def delete(self, key):
        """
        Remove an item from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            bool: True if the key was removed, False if it didn't exist
        """
        async with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                return True
            return False

    
def generate_cache_key(model_type: str, app_id: str, user_id: str, model_id: str):
    """
    Generate a cache key for a given model type, app ID, user ID, and model ID.

    Args:
        model_type (str): The type of model to generate a cache key for (e.g. "collection", "document", "message", "session", "user")
        app_id (str): The ID of the app to generate a cache key for
        user_id (str): The ID of the user to generate a cache key for
        model_id (str): The ID of the model to generate a cache key for

    Returns:
        str: A cache key for the given model type, app ID, user ID, and model ID
    """
    return f"{model_type}:{app_id}:{user_id}:{model_id}"