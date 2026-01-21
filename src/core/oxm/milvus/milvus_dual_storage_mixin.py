"""
Milvus Dual Storage Mixin

Elegant minimal-intrusion solution for Milvus dual storage:
- Zero Repository code changes (not even append/get_by_id methods)
- Compatible with main branch CRUD updates
- Dual storage completely transparent
- Same design philosophy as MongoDB DualStorageMixin

Working principle:
1. In __init__, replace self.collection with MilvusCollectionProxy
2. Proxy intercepts all Milvus calls (insert, upsert, delete, etc.)
3. Automatically handles dual storage synchronization
4. Milvus stores Lite data (vector + index fields + metadata)
5. KV-Storage stores Full data (complete entity dict)

Usage example:
    @repository("episodic_memory_milvus_repository", primary=False)
    class EpisodicMemoryMilvusRepository(
        MilvusDualStorageMixin,  # Just add Milvus Dual Storage Mixin
        BaseMilvusRepository[EpisodicMemoryCollection]
    ):
        # All code remains completely unchanged
        pass

Design advantages:
1. **Zero Repository code changes**: Just inherit from Milvus DualStorageMixin
2. **Automatic synchronization**: Proxy automatically intercepts write operations
3. **Compatible with future updates**: Main branch updates CRUD without sync needed
4. **Completely transparent**: Business code unaware of dual storage
5. **Consistent with MongoDB**: Same Mixin pattern as DualStorageMixin
"""

from typing import Optional, Set

from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
    KVStorageInterface,
)
from core.oxm.milvus.milvus_dual_storage_collection_proxy import MilvusCollectionProxy

logger = get_logger(__name__)


class MilvusDualStorageMixin:
    """
    Milvus Dual Storage Mixin

    Automatically implements dual storage by intercepting self.collection.

    Working flow:
    1. __init__ replaces self.collection with MilvusCollectionProxy
    2. Proxy intercepts insert(), upsert(), delete() methods
    3. All Milvus operations automatically sync to KV-Storage
    4. Milvus stores Lite data, KV stores Full data

    Advantages:
    - Repository code completely unchanged
    - Compatible with main branch updates
    - Dual storage logic completely transparent

    Relationship with MongoDB DualStorageMixin:
    - MongoDB: Intercepts self.model (Document class)
    - Milvus: Intercepts self.collection (AsyncCollection)
    - Same design philosophy, different interception points
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize mixin and setup dual storage interception

        Automatically:
        1. Get KV-Storage instance
        2. Replace self.collection with MilvusCollectionProxy
        3. Proxy handles all dual storage logic
        """
        super().__init__(*args, **kwargs)

        # Immediately initialize dual storage (no delay)
        self._kv_storage: Optional[KVStorageInterface] = None
        self._setup_milvus_dual_storage()

    def _setup_milvus_dual_storage(self):
        """
        Setup Milvus dual storage interception immediately

        Sets up interception at __init__ time
        """
        try:
            # 1. Get KV-Storage instance
            self._kv_storage = self._get_kv_storage()

            # 2. Get collection name (from model or collection)
            collection_name = self._get_collection_name()

            # 3. Define Lite fields (can be customized by subclass)
            lite_fields = self._get_lite_fields()

            # 4. Replace self.collection with MilvusCollectionProxy
            original_collection = self.collection
            self.collection = MilvusCollectionProxy(
                original_collection=original_collection,
                kv_storage=self._kv_storage,
                collection_name=collection_name,
                lite_fields=lite_fields,
            )

            logger.debug(
                f"✅ Milvus dual storage initialized for {collection_name}, "
                f"Repository: {self.__class__.__name__}"
            )

        except Exception as e:
            logger.error(
                f"❌ Failed to initialize Milvus dual storage for {self.__class__.__name__}: {e}",
                exc_info=True,
            )
            # Don't raise - allow Repository to work without dual storage if setup fails
            logger.warning(
                "⚠️  Milvus dual storage setup failed, Repository will work without dual storage"
            )

    def _get_kv_storage(self) -> KVStorageInterface:
        """Lazy load KV-Storage instance from DI container"""
        if self._kv_storage is None:
            from core.di import get_bean_by_type

            self._kv_storage = get_bean_by_type(KVStorageInterface)
        return self._kv_storage

    def _get_collection_name(self) -> str:
        """
        Get Milvus collection base name (for KV key prefix)

        Prefer base name without tenant suffix for logical KV keys.

        Try in order:
        1. self.model._COLLECTION_NAME (base name without suffix)
        2. self.collection.collection.name (full name with suffix)
        3. self.model.collection_name (if exists)
        4. self.model.__name__.lower() (fallback)
        """
        try:
            # Try to get base collection name (without tenant suffix)
            if hasattr(self, "model") and hasattr(self.model, "_COLLECTION_NAME"):
                return self.model._COLLECTION_NAME

            # Try AsyncCollection name (may have suffix)
            if hasattr(self, "collection") and hasattr(self.collection, "collection"):
                if hasattr(self.collection.collection, "name"):
                    return self.collection.collection.name

            # Try model collection_name
            if hasattr(self, "model") and hasattr(self.model, "collection_name"):
                return self.model.collection_name

            # Fallback: use model name
            if hasattr(self, "model"):
                return self.model.__name__.lower()

            # Last resort
            return self.__class__.__name__.lower().replace("repository", "")

        except Exception as e:
            logger.warning(f"Failed to get collection name: {e}")
            return "unknown_collection"

    def _get_lite_fields(self) -> Optional[Set[str]]:
        """
        Get Lite fields to keep in Milvus

        Subclasses can override this method to customize Lite fields.

        Default: Use MilvusCollectionProxy._default_lite_fields()
                 Includes: id, vector, user_id, group_id, event_type,
                          timestamp, episode, search_content, metadata, etc.

        Returns:
            Set of field names, or None to keep all fields

        Example override:
            def _get_lite_fields(self) -> Set[str]:
                return {
                    "id", "vector",
                    "user_id", "group_id",
                    "timestamp", "episode", "metadata"
                }
        """
        # Default: None means use MilvusCollectionProxy's default
        return None

    # Note: load_full_data_from_kv() method removed
    # Full data is now automatically loaded by MilvusCollectionProxy.search()/query()
    # Users don't need to manually load Full data anymore


# Export
__all__ = ["MilvusDualStorageMixin"]
