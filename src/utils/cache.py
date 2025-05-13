from collections import OrderedDict
import threading

class LRUCache:
    """
    A simple thread-safe LRU cache implementation.
    
    This cache automatically evicts the least recently used items when it reaches
    its capacity limit.
    """
    
    def __init__(self, capacity=100):
        """
        Initialize the LRU cache.
        
        Args:
            capacity (int): Maximum number of items to store in the cache
        """
        self._cache = OrderedDict()
        self._capacity = max(1, capacity)
        self._lock = threading.RLock()
        
    
    def get(self, key):
        """
        Get a value from the cache.
        
        Args:
            key: The key to look up
            
        Returns:
            The cached value or None if not found
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            # Move accessed item to the end to mark as most recently used
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
    
    def put(self, key, value):
        """
        Add or update an entry in the cache.
        
        Args:
            key: The key to store
            value: The value to store

        Returns:
            The value that was stored
        """
        with self._lock:
            # If key already exists, remove it first to update its position
            if key in self._cache:
                self._cache.pop(key)
            
            # If at capacity, remove the least recently used item
            elif len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)
            
            # Add the new item
            self._cache[key] = value
            return value
    
    def delete(self, key):
        """
        Remove an item from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            bool: True if the key was removed, False if it didn't exist
        """
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                return True
            return False

    
def generate_cache_key(model_type: str, app_id: str, user_id: str, model_id: str):
    return f"{model_type}:{app_id}:{user_id}:{model_id}"