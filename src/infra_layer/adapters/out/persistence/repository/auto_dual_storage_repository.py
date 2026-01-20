"""
Auto Dual Storage Repository Base Class

Automatically handles dual storage (MongoDB + KV-Storage) using reflection.
Minimizes code changes needed when upstream models are updated.

Key Features:
- Automatic indexed field extraction from Lite model using reflection
- Generic CRUD methods that work for any Full/Lite model pair
- Automatic conversion between Full and Lite models
- Minimal code in concrete repository implementations

Usage:
    class EpisodicMemoryRawRepository(AutoDualStorageRepository[EpisodicMemory, EpisodicMemoryLite]):
        def __init__(self):
            super().__init__(
                full_model=EpisodicMemory,
                lite_model=EpisodicMemoryLite,
                model_name="EpisodicMemory"
            )
"""

from typing import TypeVar, Generic, List, Optional, Dict, Any, Type
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from pydantic import BaseModel

from core.observation.logger import get_logger
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.repository.dual_storage_helper import (
    DualStorageHelper,
)

logger = get_logger(__name__)

# Type variables for Full and Lite models
TFull = TypeVar("TFull", bound=BaseModel)
TLite = TypeVar("TLite", bound=BaseModel)


class AutoDualStorageRepository(Generic[TFull, TLite], BaseRepository[TLite]):
    """
    Automatic Dual Storage Repository Base Class

    Provides automatic dual storage management using reflection to minimize
    code changes when upstream models are updated.

    Type Parameters:
        TFull: Full model type (stored in KV-Storage)
        TLite: Lite model type (stored in MongoDB with indexes)
    """

    # Fields to exclude from Lite conversion (these are managed separately)
    # revision_id is a Beanie internal field that should not be synced
    EXCLUDED_FIELDS = {"id", "created_at", "updated_at", "deleted_at", "revision_id"}

    def __init__(
        self,
        full_model: Type[TFull],
        lite_model: Type[TLite],
        model_name: str,
    ):
        """
        Initialize auto dual storage repository

        Args:
            full_model: Full model class (e.g., EpisodicMemory)
            lite_model: Lite model class (e.g., EpisodicMemoryLite)
            model_name: Model name for logging (e.g., "EpisodicMemory")
        """
        super().__init__(lite_model)

        self.full_model = full_model
        self.lite_model = lite_model
        self.model_name = model_name

        # Initialize dual storage helper
        self._dual_storage = DualStorageHelper[TFull, TLite](
            model_name=model_name, full_model=full_model
        )

        # Auto-extract indexed fields from Lite model using reflection
        self._indexed_fields = self._extract_indexed_fields()

        logger.info(
            f"✅ AutoDualStorageRepository initialized for {model_name}: "
            f"indexed_fields={self._indexed_fields}"
        )

    def _extract_indexed_fields(self) -> List[str]:
        """
        Automatically extract indexed field names from Lite model using reflection

        Returns:
            List of field names that should be stored in MongoDB

        Note:
            Excludes: id, created_at, updated_at, deleted_at (managed by base classes)
        """
        indexed_fields = []

        # Use Pydantic's model_fields for reflection
        for field_name in self.lite_model.model_fields.keys():
            if field_name not in self.EXCLUDED_FIELDS:
                indexed_fields.append(field_name)

        return indexed_fields

    def _to_lite(self, full_obj: TFull) -> TLite:
        """
        Automatically convert Full model to Lite model

        Uses reflection to copy only indexed fields from Full to Lite.

        Args:
            full_obj: Full model instance

        Returns:
            Lite model instance with only indexed fields

        Note:
            Audit fields (created_at/updated_at) are NOT copied.
            They will be auto-set by AuditBase during insert/update.
        """
        lite_data = {"id": full_obj.id}

        # Auto-copy indexed fields using reflection
        for field_name in self._indexed_fields:
            if hasattr(full_obj, field_name):
                lite_data[field_name] = getattr(full_obj, field_name)

        return self.lite_model(**lite_data)

    async def _to_full(self, lite_results: List[TLite]) -> List[TFull]:
        """
        Reconstruct Full models from KV-Storage

        Args:
            lite_results: List of Lite model instances from MongoDB

        Returns:
            List of Full model instances from KV-Storage
        """
        return await self._dual_storage.reconstruct_batch(lite_results)

    # ==================== Generic CRUD Methods ====================

    async def get_by_id(
        self,
        doc_id: str,
        session: Optional[AsyncClientSession] = None
    ) -> Optional[TFull]:
        """
        Generic get by ID - works for any model

        Args:
            doc_id: Document ID
            session: Optional MongoDB session

        Returns:
            Full model instance or None
        """
        try:
            # Read from MongoDB Lite
            lite_doc = await self.model.find_one(
                {"_id": ObjectId(doc_id)}, session=session
            )
            if not lite_doc:
                logger.debug(f"ℹ️  {self.model_name} not found: {doc_id}")
                return None

            # Reconstruct from KV-Storage
            full_objs = await self._to_full([lite_doc])
            return full_objs[0] if full_objs else None

        except Exception as e:
            logger.error(f"❌ Failed to get {self.model_name} by ID: {e}")
            return None

    async def append(
        self,
        obj: TFull,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[TFull]:
        """
        Generic append - works for any model

        Args:
            obj: Full model instance to append
            session: Optional MongoDB session

        Returns:
            Appended Full model with ID and audit fields set
        """
        try:
            # 1. Convert to Lite and insert into MongoDB
            lite_obj = self._to_lite(obj)
            await lite_obj.insert(session=session)

            # 2. Copy ID and audit fields back to Full model
            obj.id = lite_obj.id
            obj.created_at = lite_obj.created_at
            obj.updated_at = lite_obj.updated_at

            logger.info(f"✅ Appended {self.model_name}: {obj.id}")

            # 3. Write to KV-Storage
            success = await self._dual_storage.write_to_kv(obj)
            return obj if success else None

        except Exception as e:
            logger.error(f"❌ Failed to append {self.model_name}: {e}")
            return None

    async def update_by_id(
        self,
        doc_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[TFull]:
        """
        Generic update by ID - automatically updates indexed fields in MongoDB

        Args:
            doc_id: Document ID
            update_data: Fields to update
            session: Optional MongoDB session

        Returns:
            Updated Full model instance
        """
        try:
            # 1. Get Full object from KV-Storage
            kv_storage = self._dual_storage.get_kv_storage()
            kv_json = await kv_storage.get(key=doc_id)
            if not kv_json:
                logger.error(f"❌ {self.model_name} not found in KV: {doc_id}")
                return None

            full_obj = self.full_model.model_validate_json(kv_json)

            # 2. Apply updates to Full object
            for key, value in update_data.items():
                if hasattr(full_obj, key):
                    setattr(full_obj, key, value)

            # 3. Update MongoDB Lite (only indexed fields)
            lite_doc = await self.model.find_one({"_id": ObjectId(doc_id)}, session=session)
            if not lite_doc:
                logger.error(f"❌ {self.model_name} not found in MongoDB: {doc_id}")
                return None

            # Auto-update indexed fields that are present in update_data
            updated_count = 0
            for field in self._indexed_fields:
                if field in update_data and hasattr(full_obj, field):
                    setattr(lite_doc, field, getattr(full_obj, field))
                    updated_count += 1

            # Save Lite (AuditBase will auto-update updated_at)
            await lite_doc.save(session=session)
            logger.debug(
                f"✅ Updated {updated_count} indexed fields in MongoDB: {doc_id}"
            )

            # 4. Sync updated_at back to Full object
            full_obj.updated_at = lite_doc.updated_at

            # 5. Write to KV-Storage
            success = await self._dual_storage.write_to_kv(full_obj)
            return full_obj if success else None

        except Exception as e:
            logger.error(f"❌ Failed to update {self.model_name}: {e}")
            return None

    async def delete_by_id(
        self,
        doc_id: str,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Generic delete by ID - deletes from both MongoDB and KV-Storage

        Args:
            doc_id: Document ID
            session: Optional MongoDB session

        Returns:
            True if deleted successfully
        """
        try:
            # 1. Delete from KV-Storage first
            kv_deleted = await self._dual_storage.delete_from_kv(doc_id)

            # 2. Delete from MongoDB
            result = await self.model.find(
                {"_id": ObjectId(doc_id)}, session=session
            ).delete()

            mongo_deleted = (
                result.deleted_count > 0 if hasattr(result, "deleted_count") else False
            )

            if mongo_deleted:
                logger.info(f"✅ Deleted {self.model_name}: {doc_id}")
            else:
                logger.warning(f"⚠️  {self.model_name} not found for deletion: {doc_id}")

            return kv_deleted and mongo_deleted

        except Exception as e:
            logger.error(f"❌ Failed to delete {self.model_name}: {e}")
            return False

    async def find_by_filter(
        self,
        query_filter: Dict[str, Any],
        skip: int = 0,
        limit: Optional[int] = None,
        sort_field: str = "created_at",
        sort_desc: bool = False,
        session: Optional[AsyncClientSession] = None,
    ) -> List[TFull]:
        """
        Generic query by filter - automatically reconstructs from KV-Storage

        Args:
            query_filter: MongoDB query filter
            skip: Number to skip
            limit: Max number to return
            sort_field: Field to sort by
            sort_desc: Sort descending if True
            session: Optional MongoDB session

        Returns:
            List of Full model instances
        """
        try:
            # Build query
            query = self.model.find(query_filter, session=session)

            # Sort
            if sort_desc:
                query = query.sort(f"-{sort_field}")
            else:
                query = query.sort(sort_field)

            # Paginate
            query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            # Execute
            lite_results = await query.to_list()
            logger.debug(
                f"✅ Found {len(lite_results)} {self.model_name} records in MongoDB"
            )

            # Reconstruct from KV-Storage
            full_results = await self._to_full(lite_results)
            return full_results

        except Exception as e:
            logger.error(f"❌ Failed to query {self.model_name}: {e}")
            return []


__all__ = ["AutoDualStorageRepository"]
