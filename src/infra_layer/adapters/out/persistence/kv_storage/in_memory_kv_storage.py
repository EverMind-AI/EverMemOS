"""
In-Memory KV-Storage Implementation

Simple in-memory implementation for testing and development.
Not suitable for production use.
"""

from typing import Optional, Dict, List
from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
    KVStorageInterface,
)


class InMemoryKVStorage(KVStorageInterface):
    """In-memory KV-Storage implementation using dict"""

    def __init__(self):
        self._storage: Dict[str, str] = {}

    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        return self._storage.get(key)

    async def put(self, key: str, value: str) -> bool:
        """Store key-value pair"""
        self._storage[key] = value
        return True

    async def delete(self, key: str) -> bool:
        """Delete by key"""
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    async def batch_get(self, keys: List[str]) -> Dict[str, str]:
        """Batch get values"""
        result = {}
        for key in keys:
            if key in self._storage:
                result[key] = self._storage[key]
        return result

    async def batch_delete(self, keys: List[str]) -> int:
        """Batch delete keys"""
        count = 0
        for key in keys:
            if key in self._storage:
                del self._storage[key]
                count += 1
        return count


__all__ = ["InMemoryKVStorage"]
