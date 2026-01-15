"""
EventLogRecord Repository

Provides CRUD operations and query capabilities for generic event logs.
"""

from datetime import datetime
from typing import List, Optional, Type, TypeVar, Union
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
    EventLogRecordProjection,
)

# Define generic type variable
T = TypeVar('T', EventLogRecord, EventLogRecordProjection)

logger = get_logger(__name__)


@repository("event_log_record_repository", primary=True)
class EventLogRecordRawRepository(BaseRepository[EventLogRecord]):
    """
    Personal event log raw data repository

    Provides CRUD operations and basic query functions for personal event logs.
    Note: Vectors should be generated during extraction; this Repository is not responsible for vector generation.
    """

    def __init__(self):
        super().__init__(EventLogRecord)

    # ==================== Basic CRUD Methods ====================

    async def save(
        self, event_log: EventLogRecord, session: Optional[AsyncClientSession] = None
    ) -> Optional[EventLogRecord]:
        """
        Save personal event log

        Args:
            event_log: Personal event log object
            session: Optional MongoDB session, for transaction support

        Returns:
            Saved EventLogRecord or None
        """
        try:
            await event_log.insert(session=session)
            logger.info(
                "✅ Saved personal event log successfully: id=%s, user_id=%s, parent_type=%s, parent_id=%s",
                event_log.id,
                event_log.user_id,
                event_log.parent_type,
                event_log.parent_id,
            )
            return event_log
        except Exception as e:
            logger.error("❌ Failed to save personal event log: %s", e)
            return None

    async def get_by_id(
        self,
        log_id: str,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> Optional[Union[EventLogRecord, EventLogRecordProjection]]:
        """
        Get personal event log by ID

        Args:
            log_id: Log ID
            session: Optional MongoDB session, for transaction support
            model: Returned model type, default is EventLogRecord (full version), can pass EventLogRecordShort

        Returns:
            Event log object of specified type or None
        """
        try:
            object_id = ObjectId(log_id)

            # If model is not specified, use full version
            target_model = model if model is not None else self.model

            # Determine whether to use projection based on model type
            if target_model == self.model:
                result = await self.model.find_one({"_id": object_id}, session=session)
            else:
                result = await self.model.find_one(
                    {"_id": object_id}, projection_model=target_model, session=session
                )

            if result:
                logger.debug(
                    "✅ Retrieved personal event log by ID successfully: %s (model=%s)",
                    log_id,
                    target_model.__name__,
                )
            else:
                logger.debug("ℹ️  Personal event log not found: id=%s", log_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve personal event log by ID: %s", e)
            return None

    async def get_by_parent_id(
        self,
        parent_id: str,
        parent_type: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[EventLogRecord, EventLogRecordProjection]]:
        """
        Get all event logs by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")
            session: Optional MongoDB session, for transaction support
            model: Returned model type, default is EventLogRecord (full version), can pass EventLogRecordShort

        Returns:
            List of event log objects of specified type
        """
        try:
            # If model is not specified, use full version
            target_model = model if model is not None else self.model

            # Build query filter
            query_filter = {"parent_id": parent_id}
            if parent_type:
                query_filter["parent_type"] = parent_type

            # Determine whether to use projection based on model type
            if target_model == self.model:
                query = self.model.find(query_filter, session=session)
            else:
                query = self.model.find(
                    query_filter, projection_model=target_model, session=session
                )

            results = await query.to_list()
            logger.debug(
                "✅ Retrieved event logs by parent memory ID successfully: %s (type=%s), found %d records (model=%s)",
                parent_id,
                parent_type,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve event logs by parent episodic memory ID: %s", e
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
        sort_desc: bool = True,
        session: Optional[AsyncClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[EventLogRecord, EventLogRecordProjection]]:
        """
        Get list of event logs by filters (user_id, group_id, and/or time range)

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
            limit: Limit number of returned records
            skip: Number of records to skip
            sort_desc: Whether to sort by time in descending order
            session: Optional MongoDB session, for transaction support
            model: Returned model type, default is EventLogRecord (full version), can pass EventLogRecordProjection

        Returns:
            List of event log objects of specified type
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

            # If model is not specified, use full version
            target_model = model if model is not None else self.model

            # Determine whether to use projection based on model type
            if target_model == self.model:
                query = self.model.find(filter_dict, session=session)
            else:
                query = self.model.find(
                    filter_dict, projection_model=target_model, session=session
                )

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
                "✅ Retrieved event logs successfully: user_id=%s, group_id=%s, time_range=[%s, %s), found %d records (model=%s)",
                user_id,
                group_id,
                start_time,
                end_time,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ Failed to retrieve event logs: %s", e)
            return []

    async def delete_by_id(
        self, log_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete personal event log by ID

        Args:
            log_id: Log ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            object_id = ObjectId(log_id)
            result = await self.model.find({"_id": object_id}, session=session).delete()
            success = result.deleted_count > 0 if result else False

            if success:
                logger.info("✅ Deleted personal event log successfully: %s", log_id)
            else:
                logger.warning("⚠️  Personal event log to delete not found: %s", log_id)

            return success
        except Exception as e:
            logger.error("❌ Failed to delete personal event log: %s", e)
            return False

    async def delete_by_parent_id(
        self,
        parent_id: str,
        parent_type: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Delete all event logs by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of deleted records
        """
        try:
            query_filter = {"parent_id": parent_id}
            if parent_type is not None:
                query_filter["parent_type"] = parent_type

            result = await self.model.find(query_filter, session=session).delete()
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Deleted event logs by parent memory ID successfully: %s (type=%s), deleted %d records",
                parent_id,
                parent_type,
                count,
            )
            return count
        except Exception as e:
            logger.error(
                "❌ Failed to delete event logs by parent episodic memory ID: %s", e
            )
            return 0


# Export
__all__ = ["EventLogRecordRawRepository"]
