from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory_lite import (
    EpisodicMemoryLite,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_helper import (
    DualStorageHelper,
)
from agentic_layer.vectorize_service import get_vectorize_service
from common_utils.datetime_utils import get_now_with_timezone

logger = get_logger(__name__)


@repository("episodic_memory_raw_repository", primary=True)
class EpisodicMemoryRawRepository(BaseRepository[EpisodicMemoryLite]):
    """
    Episodic memory raw data repository
    Generates vectorized text content and saves it to the database
    Provides CRUD operations and basic query functions for episodic memory.
    """

    def __init__(self):
        super().__init__(EpisodicMemoryLite)
        self._dual_storage = DualStorageHelper[EpisodicMemory, EpisodicMemoryLite](
            model_name="EpisodicMemory", full_model=EpisodicMemory
        )
        self.vectorize_service = get_vectorize_service()

    # ==================== Helper Methods ====================

    def _episodic_to_lite(self, episodic: EpisodicMemory) -> EpisodicMemoryLite:
        """
        Convert full EpisodicMemory to EpisodicMemoryLite (only indexed fields).

        Args:
            episodic: Full EpisodicMemory object

        Returns:
            EpisodicMemoryLite with only indexed fields

        Note:
            Audit fields (created_at/updated_at) are not copied here.
            They will be automatically set by AuditBase during insert/update operations.
        """
        return EpisodicMemoryLite(
            id=episodic.id,
            user_id=episodic.user_id,
            group_id=episodic.group_id,
            timestamp=episodic.timestamp,
            keywords=episodic.keywords,
            linked_entities=episodic.linked_entities,
        )

    async def _episodic_lite_to_full(
        self, results: List[EpisodicMemoryLite]
    ) -> List[EpisodicMemory]:
        """
        Reconstruct full EpisodicMemory objects from KV-Storage.

        Args:
            results: List of EpisodicMemoryLite from MongoDB query

        Returns:
            List of full EpisodicMemory objects from KV-Storage
        """
        return await self._dual_storage.reconstruct_batch(results)

    # ==================== Basic CRUD Methods ====================

    async def get_by_event_id(
        self, event_id: str, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[EpisodicMemory]:
        """
        Retrieve episodic memory by event ID and user ID

        Args:
            event_id: Event ID
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            EpisodicMemory or None
        """
        try:
            # Read directly from KV-Storage (no need to check MongoDB)
            kv_storage = self._dual_storage.get_kv_storage()

            kv_json = await kv_storage.get(key=event_id)
            if not kv_json:
                logger.debug(
                    "ℹ️  Episodic memory not found: event_id=%s, user_id=%s",
                    event_id,
                    user_id,
                )
                return None

            # Deserialize and validate user_id
            full_episodic = EpisodicMemory.model_validate_json(kv_json)
            if full_episodic.user_id != user_id:
                logger.debug(
                    "ℹ️  User mismatch for episodic memory: event_id=%s, expected_user=%s, actual_user=%s",
                    event_id,
                    user_id,
                    full_episodic.user_id,
                )
                return None

            logger.debug(
                "✅ Successfully retrieved episodic memory by event ID and user ID: %s",
                event_id,
            )
            return full_episodic
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve episodic memory by event ID and user ID: %s", e
            )
            return None

    async def get_by_event_ids(
        self,
        event_ids: List[str],
        user_id: str,
        session: Optional[AsyncClientSession] = None,
    ) -> Dict[str, EpisodicMemory]:
        """
        Batch retrieve episodic memories by event ID list and user ID

        Args:
            event_ids: List of event IDs
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Dict[str, EpisodicMemory]: Dictionary with event_id as key, for fast lookup
        """
        if not event_ids:
            return {}

        try:
            kv_storage = self._dual_storage.get_kv_storage()

            # Batch get from KV-Storage
            kv_data_dict = await kv_storage.batch_get(keys=event_ids)

            # Reconstruct and filter by user_id
            result_dict = {}
            for event_id in event_ids:
                kv_json = kv_data_dict.get(event_id)
                if kv_json:
                    try:
                        full_episodic = EpisodicMemory.model_validate_json(kv_json)
                        if not user_id or full_episodic.user_id == user_id:
                            result_dict[event_id] = full_episodic
                    except Exception as e:
                        logger.error(
                            f"❌ Failed to deserialize episodic memory: {event_id}, error: {e}"
                        )

            logger.debug(
                "✅ Successfully batch retrieved episodic memories: user_id=%s, requested %d, found %d",
                user_id,
                len(event_ids),
                len(result_dict),
            )
            return result_dict
        except Exception as e:
            logger.error("❌ Failed to batch retrieve episodic memories: %s", e)
            return {}

    async def find_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[EpisodicMemory]:
        """
        Retrieve list of episodic memories by filters (user_id, group_id, and/or time range)

        Args:
            user_id: User ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by user_id
                - None or "": Filter for null/empty values (records with user_id as None or "")
                - Other values: Exact match
            group_id: Group ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by group_id
                - None or "": Filter for null/empty values (records with group_id as None or "")
                - Other values: Exact match
            start_time: Optional start time (inclusive)
            end_time: Optional end time (exclusive)
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order
            session: Optional MongoDB session, for transaction support

        Returns:
            List of EpisodicMemory
        """
        try:
            # Build query filter
            filter_dict = {}

            # Handle time range filter
            if start_time is not None and end_time is not None:
                filter_dict["timestamp"] = {"$gte": start_time, "$lt": end_time}
            elif start_time is not None:
                filter_dict["timestamp"] = {"$gte": start_time}
            elif end_time is not None:
                filter_dict["timestamp"] = {"$lt": end_time}

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

            query = self.model.find(filter_dict, session=session)

            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully retrieved episodic memories: user_id=%s, group_id=%s, time_range=[%s, %s), found %d records",
                user_id,
                group_id,
                start_time,
                end_time,
                len(results),
            )

            # Reconstruct from KV-Storage
            full_episodics = await self._episodic_lite_to_full(results)
            return full_episodics
        except Exception as e:
            logger.error("❌ Failed to retrieve episodic memories: %s", e)
            return []

    async def append_episodic_memory(
        self,
        episodic_memory: EpisodicMemory,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[EpisodicMemory]:
        """
        Append new episodic memory

        Args:
            episodic_memory: Episodic memory object
            session: Optional MongoDB session, for transaction support

        Returns:
            Appended EpisodicMemory or None
        """

        # Synchronize vector
        if episodic_memory.episode and not episodic_memory.vector:
            try:
                vector = await self.vectorize_service.get_embedding(
                    episodic_memory.episode
                )
                episodic_memory.vector = vector.tolist()
                # Set vectorization model information
                episodic_memory.vector_model = self.vectorize_service.get_model_name()
            except Exception as e:
                logger.error("❌ Failed to synchronize vector: %s", e)

        try:
            # 1. Write EpisodicMemoryLite to MongoDB (indexed fields only)
            # Note: EpisodicMemoryLite inherits AuditBase, which will auto-set created_at/updated_at on insert
            episodic_lite = self._episodic_to_lite(episodic_memory)
            await episodic_lite.insert(session=session)

            # Copy generated ID and audit fields back to full EpisodicMemory
            # (AuditBase has set these fields automatically during insert)
            episodic_memory.id = episodic_lite.id
            episodic_memory.created_at = episodic_lite.created_at
            episodic_memory.updated_at = episodic_lite.updated_at

            logger.info(
                "✅ Successfully appended episodic memory: event_id=%s, user_id=%s",
                episodic_memory.event_id,
                episodic_memory.user_id,
            )

            # 2. Write to KV-Storage (always full EpisodicMemory)
            success = await self._dual_storage.write_to_kv(episodic_memory)
            return episodic_memory if success else None
        except Exception as e:
            logger.error("❌ Failed to append episodic memory: %s", e)
            return None

    async def delete_by_event_id(
        self, event_id: str, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete episodic memory by event ID and user ID

        Args:
            event_id: Event ID
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            # 1. First delete from KV-Storage (data source)
            kv_deleted = await self._dual_storage.delete_from_kv(event_id)

            # 2. Then delete from MongoDB (index)
            mongo_deleted = False
            object_id = ObjectId(event_id)
            result = await self.model.find(
                {"_id": object_id, "user_id": user_id}, session=session
            ).delete()

            deleted_count = (
                result.deleted_count if hasattr(result, 'deleted_count') else 0
            )
            mongo_deleted = deleted_count > 0

            if mongo_deleted:
                logger.info(
                    "✅ Successfully deleted episodic memory by event ID and user ID: %s",
                    event_id,
                )
            else:
                logger.warning(
                    "⚠️  Episodic memory to delete not found: event_id=%s, user_id=%s",
                    event_id,
                    user_id,
                )

            # Return True only if deleted from both (strict consistency)
            return kv_deleted and mongo_deleted
        except Exception as e:
            logger.error(
                "❌ Failed to delete episodic memory by event ID and user ID: %s", e
            )
            return False

    async def delete_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Delete all episodic memories by user ID

        Args:
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of deleted records
        """
        try:
            # 1. Query all event IDs first
            lite_docs = await self.model.find({"user_id": user_id}).to_list()
            event_ids = [str(doc.id) for doc in lite_docs]

            # 2. First batch delete from KV-Storage
            kv_deleted_count = 0
            if event_ids:
                kv_storage = self._dual_storage.get_kv_storage()
                try:
                    kv_deleted_count = await kv_storage.batch_delete(keys=event_ids)
                    logger.debug(
                        f"✅ KV-Storage batch delete: {kv_deleted_count} records"
                    )
                except Exception as kv_error:
                    logger.warning(f"⚠️  KV-Storage batch delete failed: {kv_error}")

            # 3. Then delete from MongoDB
            result = await self.model.find({"user_id": user_id}).delete(session=session)
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Successfully deleted episodic memories by user ID: %s, deleted %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to delete episodic memories by user ID: %s", e)
            return 0

    async def find_by_filter_paginated(
        self,
        query_filter: Optional[Dict[str, Any]] = None,
        skip: int = 0,
        limit: int = 100,
        sort_field: str = "created_at",
        sort_desc: bool = False,
    ) -> List[EpisodicMemory]:
        """
        Paginated query of EpisodicMemory by filter conditions, used for data synchronization scenarios

        Args:
            query_filter: Query filter conditions, query all if None
            skip: Number of results to skip
            limit: Limit number of returned results
            sort_field: Sort field, default is created_at
            sort_desc: Whether to sort in descending order, default False (ascending)

        Returns:
            List of EpisodicMemory
        """
        try:
            # Build query
            filter_dict = query_filter if query_filter else {}
            query = self.model.find(filter_dict)

            # Sort
            if sort_desc:
                query = query.sort(f"-{sort_field}")
            else:
                query = query.sort(sort_field)

            # Paginate
            query = query.skip(skip).limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully paginated query of EpisodicMemory: filter=%s, skip=%d, limit=%d, found %d records",
                filter_dict,
                skip,
                limit,
                len(results),
            )

            # Reconstruct from KV-Storage
            full_episodics = await self._episodic_lite_to_full(results)
            return full_episodics
        except Exception as e:
            logger.error("❌ Failed to paginate query of EpisodicMemory: %s", e)
            return []


# Export
__all__ = ["EpisodicMemoryRawRepository"]
