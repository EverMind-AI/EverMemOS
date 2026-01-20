"""
Dual Storage Helper

Helper class to manage dual storage (MongoDB + KV-Storage).
Handles writing to KV-Storage and reconstructing full objects from lite objects.
"""

from typing import TypeVar, Generic, List, Optional, Type
from pydantic import BaseModel

from core.observation.logger import get_logger
from core.di import get_bean_by_type
from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
    KVStorageInterface,
)

logger = get_logger(__name__)

TFull = TypeVar("TFull", bound=BaseModel)
TLite = TypeVar("TLite", bound=BaseModel)


class DualStorageHelper(Generic[TFull, TLite]):
    """
    Helper for managing dual storage pattern

    Type Parameters:
        TFull: Full model type (stored in KV-Storage)
        TLite: Lite model type (stored in MongoDB)
    """

    def __init__(self, model_name: str, full_model: Type[TFull]):
        """
        Initialize dual storage helper

        Args:
            model_name: Model name for logging
            full_model: Full model class
        """
        self.model_name = model_name
        self.full_model = full_model
        self._kv_storage: Optional[KVStorageInterface] = None

    def get_kv_storage(self) -> KVStorageInterface:
        """Get KV-Storage instance (lazy init)"""
        if self._kv_storage is None:
            self._kv_storage = get_bean_by_type(KVStorageInterface)
        return self._kv_storage

    async def write_to_kv(self, full_obj: TFull) -> bool:
        """
        Write full object to KV-Storage

        Args:
            full_obj: Full model instance with ID set

        Returns:
            True if successful
        """
        try:
            if not full_obj.id:
                logger.error(f"❌ Cannot write to KV: {self.model_name} has no ID")
                return False

            kv_storage = self.get_kv_storage()
            key = str(full_obj.id)
            value = full_obj.model_dump_json()

            success = await kv_storage.put(key=key, value=value)
            if success:
                logger.debug(f"✅ Wrote {self.model_name} to KV-Storage: {key}")
            else:
                logger.error(f"❌ Failed to write {self.model_name} to KV-Storage: {key}")

            return success

        except Exception as e:
            logger.error(f"❌ Error writing {self.model_name} to KV-Storage: {e}")
            return False

    async def delete_from_kv(self, doc_id: str) -> bool:
        """
        Delete from KV-Storage

        Args:
            doc_id: Document ID

        Returns:
            True if deleted
        """
        try:
            kv_storage = self.get_kv_storage()
            success = await kv_storage.delete(key=doc_id)
            if success:
                logger.debug(f"✅ Deleted {self.model_name} from KV-Storage: {doc_id}")
            return success

        except Exception as e:
            logger.error(f"❌ Error deleting {self.model_name} from KV-Storage: {e}")
            return False

    async def reconstruct_batch(self, lite_objs: List[TLite]) -> List[TFull]:
        """
        Reconstruct full objects from KV-Storage

        Args:
            lite_objs: List of lite objects from MongoDB

        Returns:
            List of full objects from KV-Storage
        """
        if not lite_objs:
            return []

        try:
            kv_storage = self.get_kv_storage()
            keys = [str(obj.id) for obj in lite_objs]

            # Batch get from KV-Storage
            kv_data_dict = await kv_storage.batch_get(keys=keys)

            # Reconstruct full objects
            full_objs = []
            for lite_obj in lite_objs:
                key = str(lite_obj.id)
                kv_json = kv_data_dict.get(key)

                if kv_json:
                    try:
                        full_obj = self.full_model.model_validate_json(kv_json)
                        full_objs.append(full_obj)
                    except Exception as e:
                        logger.error(
                            f"❌ Failed to deserialize {self.model_name} from KV: {key}, {e}"
                        )
                else:
                    logger.warning(
                        f"⚠️  {self.model_name} not found in KV-Storage: {key}"
                    )

            logger.debug(
                f"✅ Reconstructed {len(full_objs)}/{len(lite_objs)} {self.model_name} from KV-Storage"
            )
            return full_objs

        except Exception as e:
            logger.error(f"❌ Error reconstructing {self.model_name} batch: {e}")
            return []


__all__ = ["DualStorageHelper"]
