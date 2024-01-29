from collections import OrderedDict

class LRUCache:
    """
    An implementation of a basic LRUcache that utilizes the built
    in OrderedDict data structure.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        """Get a value from the cache"""
        if key not in self.cache:
            return None

        # Move the accessed key to the end to indicate it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value):
        """Put a value in the cache"""
        if key in self.cache:
            # If the key already exists, move it to the end and update the value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # If the cache is full, remove the least recently used key-value pair (the first item in the OrderedDict)
                self.cache.popitem(last=False)

        # Add or update the key-value pair at the end of the OrderedDict
        self.cache[key] = value
