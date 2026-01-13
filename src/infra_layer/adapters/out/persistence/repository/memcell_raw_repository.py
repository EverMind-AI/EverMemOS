"""
MemCell Native CRUD Repository

Native data access layer for MemCell based on Beanie ODM, providing complete CRUD operations.
Does not depend on domain layer interfaces, directly operates on MemCell document models.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Type
from bson import ObjectId
from pydantic import BaseModel
from beanie.operators import And, GTE, LT, Eq, RegEx, Or
from pymongo.asynchronous.client_session import AsyncClientSession
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.di import get_bean_by_type
from core.oxm.mongo.base_repository import BaseRepository
from common_utils.datetime_utils import get_now_with_timezone

from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
)
from infra_layer.adapters.out.persistence.document.memory.memcell_lite import (
    MemCellLite,
)
from infra_layer.adapters.out.persistence.kv_storage import MemCellKVStorage

logger = get_logger(__name__)


@repository("memcell_raw_repository", primary=True)
class MemCellRawRepository(BaseRepository[MemCellLite]):
    """
    MemCell Native CRUD Repository

    Provides direct database operations on MemCell documents, including:
    - Basic CRUD operations (inherited from BaseRepository)
    - Composite queries and filtering
    - Batch operations
    - Statistics and aggregation queries
    - Transaction management (inherited from BaseRepository)
    """

    def __init__(self):
        """Initialize repository"""
        # Always use MemCellLite model for MongoDB (only indexed fields)
        super().__init__(MemCellLite)

        # Inject KV-Storage with graceful degradation
        self._kv_storage: Optional[MemCellKVStorage] = None
        try:
            self._kv_storage = get_bean_by_type(MemCellKVStorage)
            logger.info("✅ MemCell KV-Storage initialized successfully")
        except Exception as e:
            logger.error(
                f"⚠️  MemCell KV-Storage not available: {e}. "
            )
            raise e

    def _get_kv_storage(self) -> Optional[MemCellKVStorage]:
        """
        Get KV-Storage instance with availability check.

        Returns:
            KV-Storage instance if available, None otherwise
        """
        if self._kv_storage is None:
            logger.debug("KV-Storage not available, skipping KV operations")
            raise Exception("KV-Storage not available")
        return self._kv_storage

    def _memcell_to_lite(self, memcell: MemCell) -> MemCellLite:
        """
        Convert full MemCell to MemCellLite (only indexed fields)

        Args:
            memcell: Full MemCell instance

        Returns:
            MemCellLite instance with only indexed/query fields
        """
        return MemCellLite(
            id=memcell.id,
            user_id=memcell.user_id,
            timestamp=memcell.timestamp,
            group_id=memcell.group_id,
            participants=memcell.participants,
            type=memcell.type,
            keywords=memcell.keywords,
            created_at=memcell.created_at,
            updated_at=memcell.updated_at,
        )

    async def _process_query_results(
        self, results: List[MemCellLite]
    ) -> List[MemCell]:
        """
        Process query results - ALWAYS reconstruct from KV-Storage

        MongoDB is only used for querying and getting _id list.
        Actual data MUST be read from KV-Storage.

        Args:
            results: List of MongoDB query results (MemCellLite, only for _id)

        Returns:
            List of full MemCell instances from KV-Storage
        """
        if not results:
            return []

        # Get KV-Storage instance
        kv_storage = self._get_kv_storage()
        if not kv_storage:
            logger.error("❌ KV-Storage unavailable, cannot reconstruct MemCells")
            return []

        # Extract event IDs from MongoDB results
        kv_keys = [str(r.id) for r in results]

        # Batch get from KV-Storage (this is the source of truth)
        kv_data_dict = await kv_storage.batch_get(keys=kv_keys)

        # Reconstruct full MemCells from KV-Storage
        full_memcells = []
        for result in results:
            event_id = str(result.id)
            kv_json = kv_data_dict.get(event_id)
            if kv_json:
                try:
                    full_memcell = MemCell.model_validate_json(kv_json)
                    full_memcells.append(full_memcell)
                except Exception as e:
                    logger.error(f"❌ Failed to deserialize MemCell from KV: {event_id}, error: {e}")
            else:
                logger.warning(f"⚠️  MemCell not found in KV-Storage: {event_id}")

        logger.debug(f"✅ Reconstructed {len(full_memcells)}/{len(results)} MemCells from KV-Storage")

        return full_memcells

    async def get_by_event_id(self, event_id: str) -> Optional[MemCell]:
        """
        Get MemCell by event_id

        Args:
            event_id: Event ID

        Returns:
            MemCell instance or None
        """
        try:
            # Get KV-Storage instance
            kv_storage = self._get_kv_storage()
            if not kv_storage:
                logger.error("❌ KV-Storage unavailable, cannot retrieve MemCell: %s", event_id)
                return None

            # Read directly from KV-Storage (no need to check MongoDB)
            kv_json = await kv_storage.get(key=event_id)
            if not kv_json:
                logger.debug(f"⚠️  MemCell not found in KV-Storage: {event_id}")
                return None

            # Deserialize full MemCell from KV-Storage
            full_memcell = MemCell.model_validate_json(kv_json)
            logger.debug(f"✅ Retrieved MemCell from KV-Storage: {event_id}")

            return full_memcell

        except Exception as e:
            logger.error(f"❌ Failed to retrieve MemCell by event_id: {event_id}, error: {e}")
            return None

    async def get_by_event_ids(
        self, event_ids: List[str], projection_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, MemCell]:
        """
        Batch get MemCell by event_id list

        Args:
            event_ids: List of event IDs
            projection_model: Pydantic projection model class, used to specify returned fields
                             For example: pass a Pydantic model containing only specific fields
                             None means return complete MemCell objects

        Returns:
            Dict[event_id, MemCell | ProjectionModel]: Mapping dictionary from event_id to MemCell (or projection model)
            Unfound event_ids will not appear in the dictionary
        """
        try:
            if not event_ids:
                logger.debug("⚠️  event_ids list is empty, returning empty dictionary")
                return {}

            # Get KV-Storage instance
            kv_storage = self._get_kv_storage()
            if not kv_storage:
                logger.error("❌ KV-Storage unavailable, cannot retrieve MemCells")
                return {}

            # Batch get directly from KV-Storage (no need to check MongoDB)
            kv_data_dict = await kv_storage.batch_get(keys=event_ids)

            # Reconstruct full MemCells from KV-Storage
            result_dict = {}
            for event_id in event_ids:
                kv_json = kv_data_dict.get(event_id)
                if kv_json:
                    try:
                        full_memcell = MemCell.model_validate_json(kv_json)
                        result_dict[event_id] = full_memcell
                    except Exception as e:
                        logger.error(f"❌ Failed to deserialize MemCell from KV: {event_id}, error: {e}")
                else:
                    logger.debug(f"⚠️  MemCell not found in KV-Storage: {event_id}")

            logger.debug(
                f"✅ Batch retrieved {len(result_dict)}/{len(event_ids)} MemCells from KV-Storage"
            )

            return result_dict

        except Exception as e:
            logger.error("❌ Failed to batch retrieve MemCell by event_ids: %s", e)
            return {}

    async def append_memcell(
        self, memcell: MemCell, session: Optional[AsyncClientSession] = None
    ) -> Optional[MemCell]:
        """
        Append MemCell
        """
        try:
            # 1. Write MemCellLite to MongoDB (indexed fields only)
            memcell_lite = self._memcell_to_lite(memcell)
            await memcell_lite.insert(session=session)

            # Copy generated ID and audit fields back to full MemCell
            memcell.id = memcell_lite.id
            memcell.created_at = memcell_lite.created_at
            memcell.updated_at = memcell_lite.updated_at
            logger.info(f"✅ MemCell appended to MongoDB: {memcell.event_id}")

            # 2. Write to KV-Storage (always full MemCell)
            kv_storage = self._get_kv_storage()
            if kv_storage:
                try:
                    json_value = memcell.model_dump_json(by_alias=True, exclude_none=False)
                    success = await kv_storage.put(
                        key=str(memcell.id),
                        value=json_value
                    )
                    if success:
                        logger.debug(f"✅ KV-Storage write success: {memcell.event_id}")
                    else:
                        logger.error(f"⚠️  KV-Storage write failed for {memcell.event_id}")
                        return None
                except Exception as kv_error:
                    logger.error(f"⚠️  KV-Storage write error for {memcell.event_id}: {kv_error}")
                    return None

            return memcell
        except Exception as e:
            logger.error("❌ Failed to append MemCell: %s", e)
            return None

    async def update_by_event_id(
        self,
        event_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[MemCell]:
        """
        Update MemCell by event_id

        Args:
            event_id: Event ID
            update_data: Dictionary of update data
            session: Optional MongoDB session, for transaction support

        Returns:
            Updated MemCell instance or None
        """
        try:
            # 1. Get current MemCell from KV-Storage (for returning updated data)
            memcell = await self.get_by_event_id(event_id)
            if not memcell:
                logger.warning(f"⚠️  MemCell not found for update: {event_id}")
                return None

            # 2. Apply updates to full MemCell
            for key, value in update_data.items():
                if hasattr(memcell, key):
                    setattr(memcell, key, value)

            # Manually update the updated_at timestamp
            # (memcell is a plain Pydantic object from KV, not a Beanie document)
            memcell.updated_at = get_now_with_timezone()

            # 3. Update MongoDB (only indexed fields in MemCellLite)
            memcell_lite = self._memcell_to_lite(memcell)

            # Find and update lite document
            lite_doc = await MemCellLite.find_one({"_id": ObjectId(event_id)})
            if not lite_doc:
                logger.error(f"❌ MongoDB record not found for update: {event_id}")
                return None

            # Define indexed fields that can be updated
            indexed_fields = ['user_id', 'timestamp', 'group_id', 'participants', 'type', 'keywords']

            # Only update indexed fields that are present in update_data
            updated_count = 0
            for field in indexed_fields:
                if field in update_data:
                    setattr(lite_doc, field, getattr(memcell_lite, field))
                    updated_count += 1

            # Sync the same updated_at timestamp to MongoDB
            lite_doc.updated_at = memcell.updated_at

            await lite_doc.save(session=session)
            logger.debug(f"✅ Updated {updated_count} indexed fields in MongoDB: {event_id}")

            # 4. Update KV-Storage (always full MemCell)
            kv_storage = self._get_kv_storage()
            if kv_storage:
                try:
                    json_value = memcell.model_dump_json(by_alias=True, exclude_none=False)
                    await kv_storage.put(key=event_id, value=json_value)
                    logger.debug(f"✅ KV-Storage update success: {event_id}")
                except Exception as kv_error:
                    logger.warning(f"⚠️  KV-Storage update failed for {event_id}: {kv_error}")

            return memcell
        except Exception as e:
            logger.error("❌ Failed to update MemCell: %s", e)
            return None

    async def delete_by_event_id(
        self,
        event_id: str,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Soft delete MemCell by event_id

        Args:
            event_id: Event ID
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            memcell = await self.get_by_event_id(event_id)
            if memcell:
                # 1. Delete from MongoDB (primary operation)
                await memcell.delete(deleted_by=deleted_by, session=session)
                logger.debug(
                    "✅ Successfully soft deleted MemCell by event_id: %s", event_id
                )

                # 2. Delete from KV-Storage (secondary operation)
                kv_storage = self._get_kv_storage()
                if kv_storage:
                    try:
                        success = await kv_storage.delete(key=event_id)
                        if success:
                            logger.debug(f"✅ KV-Storage delete success: {event_id}")
                        else:
                            logger.warning(f"⚠️  KV-Storage delete returned False: {event_id}")
                    except Exception as kv_error:
                        logger.warning(
                            f"⚠️  KV-Storage delete failed for {event_id}: {kv_error}",
                            exc_info=True
                        )

                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to soft delete MemCell by event_id: %s", e)
            return False

    async def hard_delete_by_event_id(
        self, event_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Hard delete (physical deletion) MemCell by event_id

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Args:
            event_id: Event ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            # 1. Try to delete from MongoDB
            mongo_exists = False
            memcell = await MemCellLite.find_one({"_id": ObjectId(event_id)})
            if memcell:
                await memcell.delete(session=session)
                mongo_exists = True
                logger.debug("✅ MemCell deleted from MongoDB: %s", event_id)
            else:
                logger.warning(f"⚠️  MemCell not found in MongoDB: {event_id}, will try to delete from KV")

            # 2. Always try to delete from KV-Storage (regardless of MongoDB result)
            kv_deleted = False
            kv_storage = self._get_kv_storage()
            if kv_storage:
                try:
                    kv_deleted = await kv_storage.delete(key=event_id)
                    if kv_deleted:
                        logger.debug(f"✅ KV-Storage delete success: {event_id}")
                    else:
                        logger.debug(f"⚠️  KV-Storage key not found: {event_id}")
                except Exception as kv_error:
                    logger.warning(f"⚠️  KV-Storage delete failed: {kv_error}")

            # Return True if deleted from either storage
            return mongo_exists or kv_deleted
        except Exception as e:
            logger.error("❌ Failed to hard delete MemCell by event_id: %s", e)
            return False

    async def find_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        Query MemCell by user ID

        Args:
            user_id: User ID
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order

        Returns:
            List of MemCell
        """
        try:
            # Build query
            query = MemCellLite.find({"user_id": user_id})

            # Apply sorting
            sort_order = "-timestamp" if sort_desc else "timestamp"
            query = query.sort(sort_order)

            # Apply pagination
            if skip is not None:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            results = await query.to_list()
            logger.debug(f"✅ Found {len(results)} MemCell for user: {user_id}")

            full_memcells = await self._process_query_results(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to query MemCell by user ID: %s", e)
            return []

    async def find_by_user_and_time_range(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        Query MemCell by user ID and time range

        Args:
            user_id: User ID
            start_time: Start time
            end_time: End time
            limit: Limit number of returned results
            skip: Number of results to skip

        Returns:
            List of MemCell
        """
        try:
            # Build query with time range
            query_filter = And(
                Eq(MemCellLite.user_id, user_id),
                GTE(MemCellLite.timestamp, start_time),
                LT(MemCellLite.timestamp, end_time),
            )
            query = MemCellLite.find(query_filter)

            # Apply sorting
            sort_order = "-timestamp" if sort_desc else "timestamp"
            query = query.sort(sort_order)

            # Apply pagination
            if skip is not None:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            results = await query.to_list()
            logger.debug(
                f"✅ Found {len(results)} MemCell for user {user_id} in time range"
            )

            full_memcells = await self._process_query_results(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to query MemCell by user and time range: %s", e)
            return []

    async def find_by_group_id(
        self,
        group_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        Query MemCell by group ID

        Args:
            group_id: Group ID
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order

        Returns:
            List of MemCell
        """
        try:
            # Build query
            query = MemCellLite.find({"group_id": group_id})

            # Apply sorting
            sort_order = "-timestamp" if sort_desc else "timestamp"
            query = query.sort(sort_order)

            # Apply pagination
            if skip is not None:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            results = await query.to_list()
            logger.debug(f"✅ Found {len(results)} MemCell for group: {group_id}")

            full_memcells = await self._process_query_results(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to query MemCell by group ID: %s", e)
            return []

    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
    ) -> List[MemCell]:
        """
        Query MemCell by time range

        Args:
            start_time: Start time
            end_time: End time
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order

        Returns:
            List of MemCell
        """
        try:
            # Build query with time range
            query_filter = And(
                GTE(MemCellLite.timestamp, start_time),
                LT(MemCellLite.timestamp, end_time),
            )
            query = MemCellLite.find(query_filter)

            # Apply sorting
            sort_order = "-timestamp" if sort_desc else "timestamp"
            query = query.sort(sort_order)

            # Apply pagination
            if skip is not None:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            results = await query.to_list()
            logger.debug(f"✅ Found {len(results)} MemCell in time range")

            full_memcells = await self._process_query_results(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to query MemCell by time range: %s", e)
            return []

    async def find_by_participants(
        self,
        participants: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        Query MemCell by participants

        Args:
            participants: List of participants
            match_all: Whether to match all participants (True) or any participant (False)
            limit: Limit number of returned results
            skip: Number of results to skip

        Returns:
            List of MemCell
        """
        try:
            # Build query based on match mode
            if match_all:
                # Match all participants
                query = MemCellLite.find({"participants": {"$all": participants}})
            else:
                # Match any participant
                query = MemCellLite.find({"participants": {"$in": participants}})

            # Apply sorting
            query = query.sort("-timestamp")

            # Apply pagination
            if skip is not None:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            results = await query.to_list()
            logger.debug(f"✅ Found {len(results)} MemCell for participants")

            full_memcells = await self._process_query_results(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to query MemCell by participants: %s", e)
            return []

    async def search_by_keywords(
        self,
        keywords: List[str],
        match_all: bool = False,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[MemCell]:
        """
        Query MemCell by keywords

        Args:
            keywords: List of keywords
            match_all: Whether to match all keywords (True) or any keyword (False)
            limit: Limit number of returned results
            skip: Number of results to skip

        Returns:
            List of MemCell
        """
        try:
            # Build query based on match mode
            if match_all:
                # Match all keywords
                query = MemCellLite.find({"keywords": {"$all": keywords}})
            else:
                # Match any keyword
                query = MemCellLite.find({"keywords": {"$in": keywords}})

            # Apply sorting
            query = query.sort("-timestamp")

            # Apply pagination
            if skip is not None:
                query = query.skip(skip)
            if limit is not None:
                query = query.limit(limit)

            # Execute query
            results = await query.to_list()
            logger.debug(f"✅ Found {len(results)} MemCell matching keywords")

            full_memcells = await self._process_query_results(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to search MemCell by keywords: %s", e)
            return []

    async def delete_by_user_id(
        self,
        user_id: str,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete all MemCell of a user

        Args:
            user_id: User ID
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of soft deleted records
        """
        try:
            result = await self.model.delete_many(
                {"user_id": user_id}, deleted_by=deleted_by, session=session
            )
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully soft deleted all MemCell of user: %s, deleted %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to soft delete all MemCell of user: %s", e)
            return 0

    async def hard_delete_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Hard delete (physical deletion) all MemCell of a user

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Args:
            user_id: User ID
            session: Optional MongoDB session

        Returns:
            Number of hard deleted records
        """
        try:
            # 1. Query all event_ids to be deleted (for KV-Storage batch delete)
            event_ids = []
            kv_storage = self._get_kv_storage()
            if kv_storage:
                try:
                    memcells = await self.model.find({"user_id": user_id}).to_list()
                    event_ids = [str(mc.id) for mc in memcells]
                    logger.debug(f"Found {len(event_ids)} MemCells to delete for user: {user_id}")
                except Exception as query_error:
                    logger.warning(f"Failed to query event_ids for KV batch delete: {query_error}")

            # 2. Delete from MongoDB (primary operation)
            result = await self.model.hard_delete_many(
                {"user_id": user_id}, session=session
            )
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Successfully hard deleted all MemCell of user: %s, deleted %d records",
                user_id,
                count,
            )

            # 3. Batch delete from KV-Storage
            if event_ids:
                kv_storage = self._get_kv_storage()
                if kv_storage:
                    try:
                        deleted_count = await kv_storage.batch_delete(keys=event_ids)
                        logger.info(f"✅ KV-Storage batch delete: {deleted_count} records")
                    except Exception as kv_error:
                        logger.warning(f"⚠️  KV-Storage batch delete failed: {kv_error}")

            return count
        except Exception as e:
            logger.error("❌ Failed to hard delete all MemCell of user: %s", e)
            return 0

    async def delete_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Soft delete MemCell within time range

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of soft deleted records
        """
        try:
            filter_dict = {"timestamp": {"$gte": start_time, "$lt": end_time}}
            if user_id:
                filter_dict["user_id"] = user_id

            result = await self.model.delete_many(
                filter_dict, deleted_by=deleted_by, session=session
            )
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully soft deleted MemCell within time range: %s - %s, user: %s, deleted %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to soft delete MemCell within time range: %s", e)
            return 0

    async def hard_delete_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Hard delete (physical deletion) MemCell within time range

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of hard deleted records
        """
        try:
            filter_dict = {"timestamp": {"$gte": start_time, "$lt": end_time}}
            if user_id:
                filter_dict["user_id"] = user_id

            # 1. Query all event_ids to be deleted (for KV-Storage batch delete)
            event_ids = []
            kv_storage = self._get_kv_storage()
            if kv_storage:
                try:
                    memcells = await self.model.find(And(*conditions)).to_list()
                    event_ids = [str(mc.id) for mc in memcells]
                    logger.debug(f"Found {len(event_ids)} MemCells to delete in time range")
                except Exception as query_error:
                    logger.warning(f"Failed to query event_ids for KV batch delete: {query_error}")

            # 2. Delete from MongoDB (primary operation)
            result = await self.model.hard_delete_many(filter_dict, session=session)
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Successfully hard deleted MemCell within time range: %s - %s, user: %s, deleted %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to hard delete MemCell within time range: %s", e)
            return 0

    # ==================== Soft Delete Recovery Methods ====================

    async def restore_by_event_id(
        self, event_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Restore soft-deleted MemCell by event_id

        Args:
            event_id: Event ID

        Returns:
            Returns True if restoration succeeds, otherwise False
        """
        try:
            memcell = await self.model.hard_find_one(
                {"_id": ObjectId(event_id)}, session=session
            )
            if memcell:
                await memcell.restore()
                logger.debug(
                    "✅ Successfully restored MemCell by event_id: %s", event_id
                )
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to restore MemCell by event_id: %s", e)
            return False

    async def restore_by_user_id(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """
        Restore all soft-deleted MemCell of a user

        Args:
            user_id: User ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of restored records
        """
        try:
            result = await self.model.restore_many(
                {"user_id": user_id}, session=session
            )
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully restored all MemCell of user: %s, restored %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to restore all MemCell of user: %s", e)
            return 0

    async def restore_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> int:
        """
        Restore soft-deleted MemCell within time range

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of restored records
        """
        try:
            filter_dict = {
                "timestamp": {"$gte": start_time, "$lt": end_time},
                "deleted_at": {"$ne": None},  # Only restore deleted records
            }
            if user_id:
                filter_dict["user_id"] = user_id

            result = await self.model.restore_many(filter_dict, session=session)
            count = result.modified_count if result else 0
            logger.info(
                "✅ Successfully restored MemCell within time range: %s - %s, user: %s, restored %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to restore MemCell within time range: %s", e)
            return 0

    async def count_by_user_id(self, user_id: str) -> int:
        """
        Count number of MemCell for a user

        Args:
            user_id: User ID

        Returns:
            Number of records
        """
        try:
            count = await MemCellLite.find({"user_id": user_id}).count()
            logger.debug(f"✅ Count for user {user_id}: {count}")
            return count
        except Exception as e:
            logger.error("❌ Failed to count MemCell by user ID: %s", e)
            return 0

    async def count_by_time_range(
        self, start_time: datetime, end_time: datetime
    ) -> int:
        """
        Count number of MemCell within time range

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Number of records
        """
        try:
            query_filter = And(
                GTE(MemCellLite.timestamp, start_time),
                LT(MemCellLite.timestamp, end_time),
            )
            count = await MemCellLite.find(query_filter).count()
            logger.debug(f"✅ Count in time range: {count}")
            return count
        except Exception as e:
            logger.error("❌ Failed to count MemCell by time range: %s", e)
            return 0

    async def get_latest_by_user(
        self, user_id: str, limit: int = 10
    ) -> List[MemCell]:
        """
        Get latest MemCell records for a user

        Args:
            user_id: User ID
            limit: Limit on number of returned records

        Returns:
            List of MemCell
        """
        try:
            # Query with limit and sort
            results = await (
                MemCellLite.find({"user_id": user_id})
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(f"✅ Found {len(results)} latest MemCell for user: {user_id}")

            full_memcells = await self._process_query_results(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to get latest MemCell by user: %s", e)
            return []


# Export
__all__ = ["MemCellRawRepository"]
