"""
Dual Storage Helper - Composition Pattern

Provides common Dual Storage logic without increasing inheritance depth.
Uses composition (has-a) instead of inheritance (is-a).

Architecture:
- Lite model → MongoDB (indexed fields only)
- Full model → KV-Storage (complete data)
"""

from typing import TypeVar, Type, List, Optional, Generic
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
    KVStorageInterface,
)
from core.observation.logger import get_logger

TFull = TypeVar("TFull")
TLite = TypeVar("TLite")


class DualStorageHelper(Generic[TFull, TLite]):
    """
    Dual Storage helper class using composition pattern.

    Responsibilities:
    1. KV-Storage initialization and access
    2. Lite → Full reconstruction (batch/single)
    3. Full → KV-Storage write operations

    Usage:
        class MyRepository(BaseRepository[MyLite]):
            def __init__(self):
                super().__init__(MyLite)
                self._dual_storage = DualStorageHelper[MyFull, MyLite](
                    model_name="My",
                    full_model=MyFull
                )

            async def get_by_id(self, id: str):
                lite = await self.model.find_one({"_id": id})
                return await self._dual_storage.reconstruct_single(lite)
    """

    def __init__(self, model_name: str, full_model: Type[TFull]):
        """
        Initialize Dual Storage helper.

        Args:
            model_name: Model name for logging (e.g., "MemCell", "EpisodicMemory")
            full_model: Full model class for deserialization
        """
        self.model_name = model_name
        self.full_model = full_model
        self.logger = get_logger(__name__)
        self._kv_storage: Optional[KVStorageInterface] = None
        self._init_kv_storage()

    def _init_kv_storage(self):
        """Initialize KV-Storage with dependency injection."""
        try:
            self._kv_storage = get_bean_by_type(KVStorageInterface)
            self.logger.info(
                f"✅ {self.model_name} KV-Storage initialized successfully"
            )
        except Exception as e:
            self.logger.error(
                f"⚠️ {self.model_name} KV-Storage not available: {e}"
            )
            raise e

    def get_kv_storage(self) -> KVStorageInterface:
        """
        Get KV-Storage instance with availability check.

        Returns:
            KV-Storage instance

        Raises:
            Exception: If KV-Storage is not available
        """
        if self._kv_storage is None:
            self.logger.debug("KV-Storage not available, skipping KV operations")
            raise Exception("KV-Storage not available")
        return self._kv_storage

    async def reconstruct_batch(self, lites: List[TLite]) -> List[TFull]:
        """
        Reconstruct full objects from KV-Storage (batch).

        MongoDB is only used for querying and getting _id list.
        Actual data is read from KV-Storage.

        Args:
            lites: List of Lite objects from MongoDB query

        Returns:
            List of full objects from KV-Storage
        """
        if not lites:
            return []

        kv_storage = self.get_kv_storage()

        # Extract IDs from MongoDB results
        kv_keys = [str(lite.id) for lite in lites]

        # Batch get from KV-Storage (source of truth)
        kv_data_dict = await kv_storage.batch_get(keys=kv_keys)

        # Reconstruct full objects
        full_objects = []
        for lite in lites:
            obj_id = str(lite.id)
            kv_json = kv_data_dict.get(obj_id)
            if kv_json:
                try:
                    full_obj = self.full_model.model_validate_json(kv_json)
                    full_objects.append(full_obj)
                except Exception as e:
                    self.logger.error(
                        f"❌ Failed to deserialize {self.model_name}: {obj_id}, error: {e}"
                    )
            else:
                self.logger.warning(
                    f"⚠️ {self.model_name} not found in KV-Storage: {obj_id}"
                )

        return full_objects

    async def reconstruct_single(self, lite: Optional[TLite]) -> Optional[TFull]:
        """
        Reconstruct full object from KV-Storage (single).

        MongoDB is only used for querying and getting _id.
        Actual data is read from KV-Storage.

        Args:
            lite: Lite object from MongoDB query

        Returns:
            Full object from KV-Storage or None
        """
        if not lite:
            return None

        kv_storage = self.get_kv_storage()
        obj_id = str(lite.id)
        kv_json = await kv_storage.get(obj_id)

        if kv_json:
            try:
                full_obj = self.full_model.model_validate_json(kv_json)
                return full_obj
            except Exception as e:
                self.logger.error(
                    f"❌ Failed to deserialize {self.model_name}: {obj_id}, error: {e}"
                )
                return None
        else:
            self.logger.warning(
                f"⚠️ {self.model_name} not found in KV-Storage: {obj_id}"
            )
            return None

    async def write_to_kv(self, full_obj: TFull) -> bool:
        """
        Write full object to KV-Storage.

        Args:
            full_obj: Full object to write

        Returns:
            True if write succeeded, False otherwise
        """
        kv_storage = self.get_kv_storage()

        try:
            json_value = full_obj.model_dump_json(by_alias=True, exclude_none=False)
            success = await kv_storage.put(key=str(full_obj.id), value=json_value)

            if success:
                self.logger.debug(f"✅ KV-Storage write success: {full_obj.id}")
            else:
                self.logger.error(f"⚠️ KV-Storage write failed: {full_obj.id}")

            return success
        except Exception as kv_error:
            self.logger.error(
                f"⚠️ KV-Storage write error: {full_obj.id}: {kv_error}"
            )
            return False

    async def delete_from_kv(self, obj_id: str) -> bool:
        """
        Delete object from KV-Storage.

        Args:
            obj_id: Object ID to delete

        Returns:
            True if delete succeeded, False otherwise
        """
        kv_storage = self.get_kv_storage()

        try:
            await kv_storage.delete(obj_id)
            self.logger.debug(f"✅ KV-Storage delete success: {obj_id}")
            return True
        except Exception as kv_error:
            self.logger.error(f"⚠️ KV-Storage delete error: {obj_id}: {kv_error}")
            return False
