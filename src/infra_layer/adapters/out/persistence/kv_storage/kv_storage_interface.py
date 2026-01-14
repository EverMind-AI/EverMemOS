"""
Generic KV-Storage Interface

Abstract interface for key-value storage of memory documents (MemCell, EpisodicMemory, etc.).
Supports both in-memory (dict-based) and persistent (RocksDB/LevelDB) implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class KVStorageInterface(ABC):
    """
    Abstract interface for generic key-value storage.

    This interface defines the contract for storing and retrieving memory documents
    (MemCell, EpisodicMemory, etc.) in a key-value store. The key is the MongoDB
    ObjectId (as string), and the value is the JSON-serialized document.

    Implementations should handle errors gracefully and return None/False on failures
    rather than raising exceptions, to ensure KV-Storage issues don't break the main
    MongoDB-based workflow.
    """

    @abstractmethod
    async def put(self, key: str, value: str) -> bool:
        """
        Store a key-value pair.

        Args:
            key: Event ID (MongoDB _id as string)
            value: JSON string (from memcell.model_dump_json())

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """
        Retrieve a value by key.

        Args:
            key: Event ID (MongoDB _id as string)

        Returns:
            JSON string if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a key-value pair.

        Args:
            key: Event ID (MongoDB _id as string)

        Returns:
            True if successful (or key didn't exist), False on error
        """
        pass

    @abstractmethod
    async def batch_get(self, keys: List[str]) -> Dict[str, str]:
        """
        Retrieve multiple values by keys (batch operation).

        Args:
            keys: List of event IDs

        Returns:
            Dictionary mapping event_id to JSON string.
            Keys not found are simply not included in the result.
        """
        pass

    @abstractmethod
    async def batch_delete(self, keys: List[str]) -> int:
        """
        Delete multiple key-value pairs (batch operation).

        Args:
            keys: List of event IDs to delete

        Returns:
            Number of keys successfully deleted
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the storage connection and release resources.

        This method should be called when the storage is no longer needed,
        especially for persistent storage implementations.
        """
        pass
