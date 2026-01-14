"""
In-Memory KV-Storage Implementation

Simple dict-based implementation for generic key-value storage.
Used for MemCell, EpisodicMemory, and other memory types.
Useful for debugging and testing. Not suitable for production use
as data is lost on restart.
"""

import asyncio
from typing import Optional, List, Dict
from threading import Lock
from core.observation.logger import get_logger
from core.di.decorators import repository
from .kv_storage_interface import KVStorageInterface

logger = get_logger(__name__)


@repository("memcell_kv_storage", primary=True)
@repository("episodic_memory_kv_storage", primary=True)
class InMemoryKVStorage(KVStorageInterface):
    """
    In-memory implementation of generic KV-Storage using Python dict.

    This implementation is thread-safe and uses a simple dictionary
    to store key-value pairs in memory. All data is lost when the
    process terminates.

    Used for all memory types: MemCell, EpisodicMemory, etc.

    Suitable for:
    - Development and debugging
    - Testing
    - Small-scale deployments

    Not suitable for:
    - Production environments requiring persistence
    - Multi-process deployments
    - Large datasets
    """

    def __init__(self):
        """Initialize in-memory storage with an empty dict."""
        self._storage: Dict[str, str] = {}
        self._lock = Lock()
        logger.info("InMemoryKVStorage initialized (dict-based)")

    async def put(self, key: str, value: str) -> bool:
        """
        Store a key-value pair in memory.

        Args:
            key: Event ID
            value: JSON string

        Returns:
            True if successful, False on error
        """
        try:
            with self._lock:
                self._storage[key] = value
            logger.debug(f"InMemoryKVStorage: PUT key={key}, size={len(value)} bytes")
            return True
        except Exception as e:
            logger.error(f"InMemoryKVStorage: PUT failed for key={key}: {e}")
            return False

    async def get(self, key: str) -> Optional[str]:
        """
        Retrieve a value by key from memory.

        Args:
            key: Event ID

        Returns:
            JSON string if found, None otherwise
        """
        try:
            with self._lock:
                value = self._storage.get(key)
            if value:
                logger.debug(f"InMemoryKVStorage: GET key={key}, found size={len(value)} bytes")
            else:
                logger.debug(f"InMemoryKVStorage: GET key={key}, not found")
            return value
        except Exception as e:
            logger.error(f"InMemoryKVStorage: GET failed for key={key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """
        Delete a key-value pair from memory.

        Args:
            key: Event ID

        Returns:
            True if successful or key didn't exist, False on error
        """
        try:
            with self._lock:
                if key in self._storage:
                    del self._storage[key]
                    logger.debug(f"InMemoryKVStorage: DELETE key={key}, deleted")
                else:
                    logger.debug(f"InMemoryKVStorage: DELETE key={key}, key not found")
            return True
        except Exception as e:
            logger.error(f"InMemoryKVStorage: DELETE failed for key={key}: {e}")
            return False

    async def batch_get(self, keys: List[str]) -> Dict[str, str]:
        """
        Retrieve multiple values by keys from memory.

        Args:
            keys: List of event IDs

        Returns:
            Dictionary mapping event_id to JSON string
        """
        try:
            result = {}
            with self._lock:
                for key in keys:
                    if key in self._storage:
                        result[key] = self._storage[key]
            logger.debug(
                f"InMemoryKVStorage: BATCH_GET requested={len(keys)}, found={len(result)}"
            )
            return result
        except Exception as e:
            logger.error(f"InMemoryKVStorage: BATCH_GET failed: {e}")
            return {}

    async def batch_delete(self, keys: List[str]) -> int:
        """
        Delete multiple key-value pairs from memory.

        Args:
            keys: List of event IDs to delete

        Returns:
            Number of keys successfully deleted
        """
        try:
            deleted_count = 0
            with self._lock:
                for key in keys:
                    if key in self._storage:
                        del self._storage[key]
                        deleted_count += 1
            logger.debug(
                f"InMemoryKVStorage: BATCH_DELETE requested={len(keys)}, deleted={deleted_count}"
            )
            return deleted_count
        except Exception as e:
            logger.error(f"InMemoryKVStorage: BATCH_DELETE failed: {e}")
            return 0

    async def close(self) -> None:
        """
        Close the storage (no-op for in-memory implementation).

        In-memory storage doesn't need cleanup, but we clear the dict
        for consistency.
        """
        try:
            with self._lock:
                size = len(self._storage)
                self._storage.clear()
            logger.info(f"InMemoryKVStorage closed, cleared {size} entries")
        except Exception as e:
            logger.error(f"InMemoryKVStorage: CLOSE failed: {e}")

    def get_stats(self) -> Dict[str, int]:
        """
        Get storage statistics (useful for debugging).

        Returns:
            Dictionary with stats like entry_count, total_size
        """
        try:
            with self._lock:
                entry_count = len(self._storage)
                total_size = sum(len(v) for v in self._storage.values())
            return {"entry_count": entry_count, "total_size_bytes": total_size}
        except Exception as e:
            logger.error(f"InMemoryKVStorage: GET_STATS failed: {e}")
            return {"entry_count": 0, "total_size_bytes": 0}
