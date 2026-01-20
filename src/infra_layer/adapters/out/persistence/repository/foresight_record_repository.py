"""
ForesightRecord Repository

Provides generic CRUD operations and query capabilities for foresight records.
"""

from datetime import datetime
from typing import List, Optional, Type, TypeVar, Union
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from common_utils.datetime_utils import to_date_str
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord,
    ForesightRecordProjection,
)
from infra_layer.adapters.out.persistence.document.memory.foresight_record_lite import (
    ForesightRecordLite,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_helper import (
    DualStorageHelper,
)

# Define generic type variable
T = TypeVar('T', ForesightRecord, ForesightRecordProjection)

logger = get_logger(__name__)


@repository("foresight_record_repository", primary=True)
class ForesightRecordRawRepository(BaseRepository[ForesightRecordLite]):
    """
    Raw repository for personal foresight data

    Provides CRUD operations and basic query functions for personal foresight records.
    Note: Vectors should be generated during extraction; this Repository is not responsible for vector generation.
    """

    def __init__(self):
        super().__init__(ForesightRecordLite)
        self._dual_storage = DualStorageHelper[ForesightRecord, ForesightRecordLite](
            model_name="ForesightRecord", full_model=ForesightRecord
        )

    def _foresight_to_lite(self, foresight: ForesightRecord) -> ForesightRecordLite:
        """
        Convert full ForesightRecord to ForesightRecordLite (only indexed fields).

        Args:
            foresight: Full ForesightRecord object

        Returns:
            ForesightRecordLite with only indexed fields

        Note:
            Audit fields (created_at/updated_at) are not copied here.
            They will be automatically set by AuditBase during insert/update operations.
        """
        return ForesightRecordLite(
            id=foresight.id,
            user_id=foresight.user_id,
            group_id=foresight.group_id,
            parent_id=foresight.parent_id,
            parent_type=foresight.parent_type,
            start_time=foresight.start_time,
            end_time=foresight.end_time,
        )

    async def _foresight_lite_to_full(
        self, results: List[ForesightRecordLite]
    ) -> List[ForesightRecord]:
        """
        Reconstruct full ForesightRecord objects from KV-Storage.

        Args:
            results: List of ForesightRecordLite from MongoDB query

        Returns:
            List of full ForesightRecord objects from KV-Storage
        """
        return await self._dual_storage.reconstruct_batch(results)

    def _convert_to_projection_if_needed(
        self,
        full_foresights: List[ForesightRecord],
        target_model: Optional[Type[T]] = None,
    ) -> List[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Convert full ForesightRecord objects to Projection if needed.

        Args:
            full_foresights: List of full ForesightRecord objects
            target_model: Target model type

        Returns:
            List of converted objects
        """
        if not target_model or target_model == ForesightRecord:
            return full_foresights

        if target_model == ForesightRecordProjection:
            return [
                ForesightRecordProjection(
                    id=f.id,
                    user_id=f.user_id,
                    user_name=f.user_name,
                    group_id=f.group_id,
                    group_name=f.group_name,
                    content=f.content,
                    parent_type=f.parent_type,
                    parent_id=f.parent_id,
                    start_time=f.start_time,
                    end_time=f.end_time,
                    duration_days=f.duration_days,
                    participants=f.participants,
                    vector_model=f.vector_model,
                    evidence=f.evidence,
                    extend=f.extend,
                    created_at=f.created_at,
                    updated_at=f.updated_at,
                )
                for f in full_foresights
            ]

        return full_foresights

    # ==================== Basic CRUD Methods ====================

    async def save(
        self, foresight: ForesightRecord, session: Optional[AsyncClientSession] = None
    ) -> Optional[ForesightRecord]:
        """
        Save personal foresight record

        Args:
            foresight: ForesightRecord object
            session: Optional MongoDB session for transaction support

        Returns:
            Saved ForesightRecord or None
        """
        try:
            # 1. Write ForesightRecordLite to MongoDB (indexed fields only)
            # Note: ForesightRecordLite inherits AuditBase, which will auto-set created_at/updated_at on insert
            foresight_lite = self._foresight_to_lite(foresight)
            await foresight_lite.insert(session=session)

            # Copy generated ID and audit fields back to full ForesightRecord
            # (AuditBase has set these fields automatically during insert)
            foresight.id = foresight_lite.id
            foresight.created_at = foresight_lite.created_at
            foresight.updated_at = foresight_lite.updated_at

            logger.info(
                "✅ Saved personal foresight successfully: id=%s, user_id=%s, parent_type=%s, parent_id=%s",
                foresight.id,
                foresight.user_id,
                foresight.parent_type,
                foresight.parent_id,
            )

            # 2. Write to KV-Storage (always full ForesightRecord)
            success = await self._dual_storage.write_to_kv(foresight)
            return foresight if success else None
        except Exception as e:
            logger.error("❌ Failed to save personal foresight: %s", e)
            return None

    async def get_by_id(
        self,
        memory_id: str,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> Optional[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Retrieve personal foresight by ID

        Args:
            memory_id: Memory ID
            session: Optional MongoDB session for transaction support
            model: Type of model to return, defaults to ForesightRecord (full version)

        Returns:
            Foresight object of specified type or None
        """
        try:
            object_id = ObjectId(memory_id)

            # Query MongoDB Lite model first
            lite_result = await self.model.find_one({"_id": object_id}, session=session)
            if not lite_result:
                logger.debug("ℹ️  Personal foresight not found: id=%s", memory_id)
                return None

            # Reconstruct from KV-Storage
            full_foresights = await self._foresight_lite_to_full([lite_result])
            if not full_foresights:
                return None

            # Convert to target model type if needed
            results = self._convert_to_projection_if_needed(full_foresights, model)
            result = results[0] if results else None

            if result:
                target_model = model if model is not None else ForesightRecord
                logger.debug(
                    "✅ Retrieved personal foresight by ID successfully: %s (model=%s)",
                    memory_id,
                    target_model.__name__,
                )
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve personal foresight by ID: %s", e)
            return None

    async def get_by_parent_id(
        self,
        parent_id: str,
        parent_type: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Retrieve all foresights by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")
            session: Optional MongoDB session for transaction support
            model: Type of model to return, defaults to ForesightRecord (full version)

        Returns:
            List of foresight objects of specified type
        """
        try:
            # Build query filter
            query_filter = {"parent_id": parent_id}
            if parent_type:
                query_filter["parent_type"] = parent_type

            # Query MongoDB for Lite results
            lite_results = await self.model.find(query_filter, session=session).to_list()

            # Reconstruct from KV-Storage
            full_foresights = await self._foresight_lite_to_full(lite_results)

            # Convert to target model type if needed
            results = self._convert_to_projection_if_needed(full_foresights, model)

            target_model = model if model is not None else ForesightRecord
            logger.debug(
                "✅ Retrieved foresights by parent memory ID successfully: %s (type=%s), found %d records (model=%s)",
                parent_id,
                parent_type,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve foresights by parent episodic memory ID: %s", e
            )
            return []

    async def find_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[ForesightRecord, ForesightRecordProjection]]:
        """
        Retrieve list of foresights by filters (user_id, group_id, and/or validity time range)

        Args:
            user_id: User ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by user_id
                - None or "": Filter for null/empty values (records with user_id as None or "")
                - Other values: Exact match
            group_id: Group ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by group_id
                - None or "": Filter for null/empty values (records with group_id as None or "")
                - Other values: Exact match
            start_time: Optional query start time (datetime object)
                - Filters foresights whose validity period overlaps with [start_time, end_time)
                - Will be converted to ISO date string (YYYY-MM-DD) internally
            end_time: Optional query end time (datetime object)
                - Filters foresights whose validity period overlaps with [start_time, end_time)
                - Will be converted to ISO date string (YYYY-MM-DD) internally
            limit: Limit number of returned records
            skip: Number of records to skip
            session: Optional MongoDB session for transaction support
            model: Type of model to return, defaults to ForesightRecord (full version)

        Returns:
            List of foresight objects of specified type
        """
        try:
            # Build query filter
            filter_dict = {}

            # Convert datetime to ISO date string for foresight validity period comparison
            start_str = to_date_str(start_time)
            end_str = to_date_str(end_time)

            # Handle time range filter (overlap query)
            # Logic: foresight.start_time <= query.end_time AND foresight.end_time >= query.start_time
            if start_str is not None and end_str is not None:
                filter_dict["$and"] = [
                    {"start_time": {"$lte": end_str}},
                    {"end_time": {"$gte": start_str}},
                ]
            elif start_str is not None:
                # Only start_time: find foresights that end after start_time
                filter_dict["end_time"] = {"$gte": start_str}
            elif end_str is not None:
                # Only end_time: find foresights that start before end_time
                filter_dict["start_time"] = {"$lte": end_str}

            # Handle user_id filter
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    # Explicitly filter for null or empty string
                    filter_dict["user_id"] = {"$in": [None, ""]}
                else:
                    filter_dict["user_id"] = user_id

            # Handle group_id filter
            if group_id != MAGIC_ALL:
                if group_id == "" or group_id is None:
                    # Explicitly filter for null or empty string
                    filter_dict["group_id"] = {"$in": [None, ""]}
                else:
                    filter_dict["group_id"] = group_id

            # Query MongoDB Lite
            query = self.model.find(filter_dict, session=session)

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            lite_results = await query.to_list()

            # Reconstruct from KV-Storage
            full_foresights = await self._foresight_lite_to_full(lite_results)

            # Convert to target model type if needed
            results = self._convert_to_projection_if_needed(full_foresights, model)

            target_model = model if model is not None else ForesightRecord
            logger.debug(
                "✅ Retrieved foresights successfully: user_id=%s, group_id=%s, time_range=[%s, %s), found %d records (model=%s)",
                user_id,
                group_id,
                start_str,
                end_str,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve foresights: %s", e)
            return []

    async def delete_by_id(
        self, memory_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete personal foresight by ID

        Args:
            memory_id: Memory ID
            session: Optional MongoDB session for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            object_id = ObjectId(memory_id)

            # Delete from MongoDB
            result = await self.model.find({"_id": object_id}, session=session).delete()
            success = result.deleted_count > 0 if result else False

            if success:
                # Delete from KV-Storage
                await self._dual_storage.delete_from_kv(memory_id)
                logger.info("✅ Deleted personal foresight successfully: %s", memory_id)
            else:
                logger.warning(
                    "⚠️  Personal foresight to delete not found: %s", memory_id
                )

            return success
        except Exception as e:
            logger.error("❌ Failed to delete personal foresight: %s", e)
            return False

    async def delete_by_parent_id(
        self,
        parent_id: str,
        parent_type: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Delete all foresights by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")
            session: Optional MongoDB session for transaction support

        Returns:
            Number of deleted records
        """
        try:
            # Get all IDs first for KV-Storage deletion
            query_filter = {"parent_id": parent_id}
            if parent_type is not None:
                query_filter["parent_type"] = parent_type

            lite_results = await self.model.find(query_filter, session=session).to_list()
            memory_ids = [str(result.id) for result in lite_results]

            # Delete from MongoDB
            result = await self.model.find(query_filter, session=session).delete()
            count = result.deleted_count if result else 0

            # Delete from KV-Storage
            if count > 0 and memory_ids:
                kv_storage = self._dual_storage.get_kv_storage()
                try:
                    await kv_storage.batch_delete(memory_ids)
                except Exception as kv_error:
                    logger.error(
                        f"⚠️  KV-Storage batch delete error for parent {parent_id}: {kv_error}"
                    )

            logger.info(
                "✅ Deleted foresights by parent memory ID successfully: %s (type=%s), deleted %d records",
                parent_id,
                parent_type,
                count,
            )
            return count
        except Exception as e:
            logger.error(
                "❌ Failed to delete foresights by parent episodic memory ID: %s", e
            )
            return 0


# Export
__all__ = ["ForesightRecordRawRepository"]
