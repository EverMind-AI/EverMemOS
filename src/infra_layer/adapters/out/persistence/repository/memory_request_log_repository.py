# -*- coding: utf-8 -*-
"""
MemoryRequestLog Repository

Memory request log data access layer, providing CRUD operations for memories request records.
Used as a replacement for the conversation_data functionality.
"""

from datetime import datetime
from typing import List, Optional
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
    MemoryRequestLog,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_mixin import (
    DualStorageMixin,
)

logger = get_logger(__name__)


@repository("memory_request_log_repository", primary=True)
class MemoryRequestLogRepository(
    DualStorageMixin,  # 添加双存储支持 - 自动拦截 MongoDB 调用
    BaseRepository[MemoryRequestLog],
):
    """
    Memory Request Log Repository

    Provides CRUD operations and query functionality for memories API request records.
    Can be used as an alternative implementation for conversation_data.
    """

    def __init__(self):
        super().__init__(MemoryRequestLog)

    # ==================== Save Methods ====================

    async def save(
        self,
        memory_request_log: MemoryRequestLog,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[MemoryRequestLog]:
        """
        Save Memory request log

        Args:
            memory_request_log: MemoryRequestLog object
            session: Optional MongoDB session

        Returns:
            Saved MemoryRequestLog or None
        """
        try:
            await memory_request_log.insert(session=session)
            logger.debug(
                "Memory request log saved successfully: id=%s, group_id=%s, request_id=%s",
                memory_request_log.id,
                memory_request_log.group_id,
                memory_request_log.request_id,
            )
            return memory_request_log
        except Exception as e:
            logger.error("Failed to save Memory request log: %s", e)
            return None

    # ==================== Query Methods ====================

    async def get_by_request_id(
        self, request_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[MemoryRequestLog]:
        """
        Get Memory request log by request ID

        Args:
            request_id: Request ID
            session: Optional MongoDB session

        Returns:
            MemoryRequestLog or None
        """
        try:
            result = await self.model.find_one(
                {"request_id": request_id}, session=session
            )
            return result
        except Exception as e:
            logger.error("Failed to get Memory request log by request ID: %s", e)
            return None

    async def find_by_group_id(
        self,
        group_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        sync_status: Optional[int] = 0,
        session: Optional[AsyncClientSession] = None,
    ) -> List[MemoryRequestLog]:
        """
        Query Memory request logs by group_id

        Args:
            group_id: Conversation group ID
            start_time: Start time
            end_time: End time
            limit: Maximum number of records to return
            sync_status: Sync status filter (default 0=in window accumulation, None=no filter)
                - -1: Just a log record
                -  0: In window accumulation
                -  1: Already fully used
                - None: No filter, return all statuses
            session: Optional MongoDB session

        Returns:
            List of MemoryRequestLog
        """
        try:
            query = {"group_id": group_id}

            # Filter by status
            if sync_status is not None:
                query["sync_status"] = sync_status

            if start_time:
                query["created_at"] = {"$gte": start_time}
            if end_time:
                if "created_at" in query:
                    query["created_at"]["$lte"] = end_time
                else:
                    query["created_at"] = {"$lte": end_time}

            results = (
                await self.model.find(query, session=session)
                .sort([("created_at", 1)])  # Ascending order by time, oldest first
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "Query Memory request logs by group_id: group_id=%s, sync_status=%s, count=%d",
                group_id,
                sync_status,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("Failed to query Memory request logs by group_id: %s", e)
            return []

    async def find_by_group_id_with_statuses(
        self,
        group_id: str,
        sync_status_list: List[int],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        ascending: bool = True,
        exclude_message_ids: Optional[List[str]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> List[MemoryRequestLog]:
        """
        Query Memory request logs by group_id with multiple sync_status values

        This method is designed to handle edge cases by allowing queries for
        multiple sync_status values at once (e.g., both -1 and 0).

        Args:
            group_id: Conversation group ID
            sync_status_list: List of sync_status values to filter by
                - [-1]: Just log records
                - [0]: In window accumulation
                - [1]: Already fully used
                - [-1, 0]: Both pending and accumulating (for edge case handling)
            start_time: Start time (optional)
            end_time: End time (optional)
            limit: Maximum number of records to return
            ascending: If True (default), sort by created_at ascending (oldest first);
                       if False, sort descending (newest first)
            exclude_message_ids: Message IDs to exclude from results
            session: Optional MongoDB session

        Returns:
            List of MemoryRequestLog
        """
        try:
            query = {"group_id": group_id}

            # Filter by multiple statuses
            if sync_status_list:
                if len(sync_status_list) == 1:
                    query["sync_status"] = sync_status_list[0]
                else:
                    query["sync_status"] = {"$in": sync_status_list}

            if start_time:
                query["created_at"] = {"$gte": start_time}
            if end_time:
                if "created_at" in query:
                    query["created_at"]["$lte"] = end_time
                else:
                    query["created_at"] = {"$lte": end_time}

            # Exclude specific message_ids
            if exclude_message_ids:
                query["message_id"] = {"$nin": exclude_message_ids}

            # Determine sort order
            sort_order = 1 if ascending else -1

            results = (
                await self.model.find(query, session=session)
                .sort([("created_at", sort_order)])
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "Query Memory request logs by group_id with statuses: group_id=%s, sync_status_list=%s, exclude=%d, count=%d",
                group_id,
                sync_status_list,
                len(exclude_message_ids) if exclude_message_ids else 0,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to query Memory request logs by group_id with statuses: %s", e
            )
            return []

    async def find_by_user_id(
        self,
        user_id: str,
        limit: int = 100,
        session: Optional[AsyncClientSession] = None,
    ) -> List[MemoryRequestLog]:
        """
        Query Memory request logs by user ID

        Args:
            user_id: User ID
            limit: Maximum number of records to return
            session: Optional MongoDB session

        Returns:
            List of MemoryRequestLog
        """
        try:
            results = (
                await self.model.find({"user_id": user_id}, session=session)
                .sort([("created_at", -1)])
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "Query Memory request logs by user_id: user_id=%s, count=%d",
                user_id,
                len(results),
            )
            return results
        except Exception as e:
            logger.error("Failed to query Memory request logs by user_id: %s", e)
            return []

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Delete Memory request logs by group_id

        Args:
            group_id: Conversation group ID
            session: Optional MongoDB session

        Returns:
            Number of deleted records
        """
        try:
            result = await self.model.find(
                {"group_id": group_id}, session=session
            ).delete()
            deleted_count = result.deleted_count if result else 0
            logger.info(
                "Deleted Memory request logs: group_id=%s, deleted=%d",
                group_id,
                deleted_count,
            )
            return deleted_count
        except Exception as e:
            logger.error(
                "Failed to delete Memory request logs: group_id=%s, error=%s",
                group_id,
                e,
            )
            return 0

    # ==================== Sync Status Management ====================
    # sync_status state transitions:
    # -1 (log record) -> 0 (window accumulation) -> 1 (used)
    #
    # - save_conversation_data: -1 -> 0 (confirm enters window accumulation)
    # - delete_conversation_data: 0 -> 1 (mark as fully used)

    async def confirm_accumulation_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Confirm log records for the specified group_id as window accumulation state

        Batch update sync_status: -1 -> 0, used for save_conversation_data.
        Uses (group_id, sync_status) composite index for efficient querying.

        Note: This method updates all sync_status=-1 records under this group.
        For precise control, use confirm_accumulation_by_message_ids.

        Args:
            group_id: Conversation group ID
            session: Optional MongoDB session

        Returns:
            Number of updated records
        """
        try:
            # 1. Find documents to update (get IDs before update)
            docs_to_update = await self.model.find(
                {"group_id": group_id, "sync_status": -1}, session=session
            ).to_list()

            if not docs_to_update:
                return 0

            # 2. Update MongoDB Lite data
            collection = self.model.get_pymongo_collection()
            result = await collection.update_many(
                {"group_id": group_id, "sync_status": -1},
                {"$set": {"sync_status": 0}},
                session=session,
            )
            modified_count = result.modified_count if result else 0

            # 3. Update KV-Storage with full data
            if modified_count > 0:
                import json

                for doc in docs_to_update:
                    try:
                        kv_key = str(doc.id)
                        # Load existing full data from KV
                        kv_value = await self._kv_storage.get(key=kv_key)
                        if kv_value:
                            # Parse existing data
                            full_data = json.loads(kv_value)
                            # Update sync_status field
                            full_data["sync_status"] = 0
                            # Write back to KV
                            kv_value = json.dumps(full_data)
                            await self._kv_storage.put(key=kv_key, value=kv_value)
                        else:
                            logger.warning(f"⚠️  KV miss for {doc.id}, cannot update")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to sync to KV-Storage for {doc.id}: {e}")

            logger.info(
                "Confirmed window accumulation: group_id=%s, modified=%d",
                group_id,
                modified_count,
            )
            return modified_count
        except Exception as e:
            logger.error(
                "Failed to confirm window accumulation: group_id=%s, error=%s",
                group_id,
                e,
            )
            return 0

    async def confirm_accumulation_by_message_ids(
        self,
        group_id: str,
        message_ids: List[str],
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Confirm log records for the specified message_id list as window accumulation state

        Precise update: only update records with specified message_id to avoid
        accidentally updating data from other concurrent requests.
        sync_status: -1 -> 0

        Args:
            group_id: Conversation group ID (for additional validation)
            message_ids: List of message_ids to update
            session: Optional MongoDB session

        Returns:
            Number of updated records
        """
        if not message_ids:
            logger.debug("message_ids is empty, skipping update")
            return 0

        try:
            # 1. Find documents to update
            docs_to_update = await self.model.find(
                {
                    "group_id": group_id,
                    "message_id": {"$in": message_ids},
                    "sync_status": -1,
                },
                session=session,
            ).to_list()

            if not docs_to_update:
                return 0

            # 2. Update MongoDB Lite data
            collection = self.model.get_pymongo_collection()
            result = await collection.update_many(
                {
                    "group_id": group_id,
                    "message_id": {"$in": message_ids},
                    "sync_status": -1,
                },
                {"$set": {"sync_status": 0}},
                session=session,
            )
            modified_count = result.modified_count if result else 0

            # 3. Update KV-Storage with full data
            if modified_count > 0:
                import json

                for doc in docs_to_update:
                    try:
                        kv_key = str(doc.id)
                        # Load existing full data from KV
                        kv_value = await self._kv_storage.get(key=kv_key)
                        if kv_value:
                            # Parse existing data
                            full_data = json.loads(kv_value)
                            # Update sync_status field
                            full_data["sync_status"] = 0
                            # Write back to KV
                            kv_value = json.dumps(full_data)
                            await self._kv_storage.put(key=kv_key, value=kv_value)
                        else:
                            logger.warning(f"⚠️  KV miss for {doc.id}, cannot update")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to sync to KV-Storage for {doc.id}: {e}")

            logger.info(
                "Confirmed window accumulation (precise): group_id=%s, message_ids=%d, modified=%d",
                group_id,
                len(message_ids),
                modified_count,
            )
            return modified_count
        except Exception as e:
            logger.error(
                "Failed to confirm window accumulation (precise): group_id=%s, error=%s",
                group_id,
                e,
            )
            return 0

    async def mark_as_used_by_group_id(
        self,
        group_id: str,
        exclude_message_ids: Optional[List[str]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Mark all pending and accumulating data for the specified group_id as used

        Batch update sync_status: -1 or 0 -> 1, used for delete_conversation_data
        (after boundary detection). Processes both pending (-1) and accumulating (0) records.

        Uses (group_id, sync_status) composite index for efficient querying.

        Args:
            group_id: Conversation group ID
            exclude_message_ids: Message IDs to exclude from update
            session: Optional MongoDB session

        Returns:
            Number of updated records
        """
        try:
            # 1. Build query
            query = {"group_id": group_id, "sync_status": {"$in": [-1, 0]}}

            # Exclude specific message_ids
            if exclude_message_ids:
                query["message_id"] = {"$nin": exclude_message_ids}

            # 2. Find documents to update
            docs_to_update = await self.model.find(query, session=session).to_list()

            if not docs_to_update:
                return 0

            # 3. Update MongoDB Lite data
            collection = self.model.get_pymongo_collection()
            result = await collection.update_many(
                query, {"$set": {"sync_status": 1}}, session=session
            )
            modified_count = result.modified_count if result else 0

            # 4. Update KV-Storage with full data
            if modified_count > 0:
                import json

                for doc in docs_to_update:
                    try:
                        kv_key = str(doc.id)
                        # Load existing full data from KV
                        kv_value = await self._kv_storage.get(key=kv_key)
                        if kv_value:
                            # Parse existing data
                            full_data = json.loads(kv_value)
                            # Update sync_status field
                            full_data["sync_status"] = 1
                            # Write back to KV
                            kv_value = json.dumps(full_data)
                            await self._kv_storage.put(key=kv_key, value=kv_value)
                        else:
                            logger.warning(f"⚠️  KV miss for {doc.id}, cannot update")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to sync to KV-Storage for {doc.id}: {e}")

            logger.info(
                "Marked as used: group_id=%s, exclude=%d, modified=%d",
                group_id,
                len(exclude_message_ids) if exclude_message_ids else 0,
                modified_count,
            )
            return modified_count
        except Exception as e:
            logger.error("Failed to mark as used: group_id=%s, error=%s", group_id, e)
            return 0

    # ==================== Flexible Query Methods ====================

    async def find_pending_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        sync_status_list: Optional[List[int]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        skip: int = 0,
        ascending: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[MemoryRequestLog]:
        """
        Query pending Memory request logs by flexible filters

        Supports MAGIC_ALL logic similar to episodic_memory_raw_repository:
        - MAGIC_ALL ("__all__"): Don't filter by this field
        - None or "": Filter for null/empty values
        - Other values: Exact match

        Args:
            user_id: User ID filter
                - MAGIC_ALL: Don't filter by user_id
                - None or "": Filter for null/empty values
                - Other values: Exact match
            group_id: Group ID filter
                - MAGIC_ALL: Don't filter by group_id
                - None or "": Filter for null/empty values
                - Other values: Exact match
            sync_status_list: List of sync_status values to filter by
                - Default: [-1, 0] (pending and accumulating, i.e., unconsumed)
                - [-1]: Just log records
                - [0]: In window accumulation
                - [1]: Already fully used
            start_time: Start time (optional)
            end_time: End time (optional)
            limit: Maximum number of records to return
            skip: Number of records to skip
            ascending: If True (default), sort by created_at ascending (oldest first);
                       if False, sort descending (newest first)
            session: Optional MongoDB session

        Returns:
            List of MemoryRequestLog
        """
        # Default to unconsumed statuses
        if sync_status_list is None:
            sync_status_list = [-1, 0]

        try:
            query = {}

            # Handle user_id filter with MAGIC_ALL logic
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    # Explicitly filter for null or empty string
                    query["user_id"] = {"$in": [None, ""]}
                else:
                    query["user_id"] = user_id

            # Handle group_id filter with MAGIC_ALL logic
            if group_id != MAGIC_ALL:
                if group_id == "" or group_id is None:
                    # Explicitly filter for null or empty string
                    query["group_id"] = {"$in": [None, ""]}
                else:
                    query["group_id"] = group_id

            # Filter by sync_status
            if sync_status_list:
                if len(sync_status_list) == 1:
                    query["sync_status"] = sync_status_list[0]
                else:
                    query["sync_status"] = {"$in": sync_status_list}

            # Handle time range filter
            if start_time is not None or end_time is not None:
                time_filter = {}
                if start_time is not None:
                    time_filter["$gte"] = start_time
                if end_time is not None:
                    time_filter["$lte"] = end_time
                query["created_at"] = time_filter

            # Determine sort order
            sort_order = 1 if ascending else -1

            results = (
                await self.model.find(query, session=session)
                .sort([("created_at", sort_order)])
                .skip(skip)
                .limit(limit)
                .to_list()
            )

            logger.debug(
                "Query pending Memory request logs: user_id=%s, group_id=%s, "
                "sync_status_list=%s, skip=%d, limit=%d, count=%d",
                user_id,
                group_id,
                sync_status_list,
                skip,
                limit,
                len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "Failed to query pending Memory request logs: user_id=%s, group_id=%s, error=%s",
                user_id,
                group_id,
                e,
            )
            return []
