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
from core.oxm.mongo.base_repository import BaseRepository
from common_utils.datetime_utils import get_now_with_timezone

from infra_layer.adapters.out.persistence.document.memory.memcell import (
    MemCell,
    DataTypeEnum,
)
from infra_layer.adapters.out.persistence.document.memory.memcell_lite import (
    MemCellLite,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_helper import (
    DualStorageHelper,
)

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
        super().__init__(MemCellLite)
        self._dual_storage = DualStorageHelper[MemCell, MemCellLite](
            model_name="MemCell", full_model=MemCell
        )

    def _memcell_to_lite(self, memcell: MemCell) -> MemCellLite:
        """
        Convert full MemCell to MemCellLite (only indexed fields)

        Args:
            memcell: Full MemCell instance

        Returns:
            MemCellLite instance with only indexed/query fields

        Note:
            Audit fields (created_at/updated_at) are not copied here.
            They will be automatically set by AuditBase during insert/update operations.
        """
        return MemCellLite(
            id=memcell.id,
            user_id=memcell.user_id,
            timestamp=memcell.timestamp,
            group_id=memcell.group_id,
            participants=memcell.participants,
            type=memcell.type,
            keywords=memcell.keywords,
        )

    async def _memcell_lite_to_full(
        self, results: List[MemCellLite]
    ) -> List[MemCell]:
        """
        Reconstruct full MemCells from KV-Storage.

        Args:
            results: List of MemCellLite from MongoDB query

        Returns:
            List of full MemCell instances from KV-Storage
        """
        return await self._dual_storage.reconstruct_batch(results)

    async def get_by_event_id(self, event_id: str) -> Optional[MemCell]:
        """
        Get MemCell by event_id

        Args:
            event_id: Event ID

        Returns:
            MemCell instance or None (returns None if soft-deleted)
        """
        try:
            # First check MongoDB to see if the record is soft-deleted
            # Using find_one (not hard_find_one) will return None if soft-deleted
            memcell_lite = await self.model.find_one({"_id": ObjectId(event_id)})
            if not memcell_lite:
                logger.debug(f"⚠️  MemCell not found or soft-deleted in MongoDB: {event_id}")
                return None

            # Read from KV-Storage
            kv_storage = self._dual_storage.get_kv_storage()
            kv_json = await kv_storage.get(key=event_id)
            if not kv_json:
                logger.debug(f"⚠️  MemCell not found in KV-Storage: {event_id}")
                return None

            # Deserialize full MemCell from KV-Storage
            full_memcell = MemCell.model_validate_json(kv_json)
            logger.debug(f"✅ Retrieved MemCell from KV-Storage: {event_id}")

            return full_memcell

        except Exception as e:
            logger.error("❌ Failed to retrieve MemCell by event_id: %s", e)
            return None

    async def get_by_event_ids(
        self, event_ids: List[str], projection_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Batch get MemCell by event_id list

        Args:
            event_ids: List of event IDs
            projection_model: Pydantic projection model class, used to specify returned fields
                             For example: pass a Pydantic model containing only specific fields
                             None means return complete MemCell objects

        Returns:
            Dict[event_id, MemCell | ProjectionModel]: Mapping dictionary from event_id to MemCell (or projection model)
            Unfound event_ids will not appear in the dictionary (includes soft-deleted records)
        """
        try:
            if not event_ids:
                logger.debug("⚠️  event_ids list is empty, returning empty dictionary")
                return {}

            # First check MongoDB to filter out soft-deleted records
            # Using find (not hard_find) will exclude soft-deleted records
            object_ids = [ObjectId(eid) for eid in event_ids]
            memcell_lites = await self.model.find({"_id": {"$in": object_ids}}).to_list()

            # Create set of valid (non-deleted) event_ids
            valid_event_ids = {str(lite.id) for lite in memcell_lites}

            if not valid_event_ids:
                logger.debug("⚠️  No valid (non-deleted) MemCells found in MongoDB")
                return {}

            # Batch get from KV-Storage (only for valid event_ids)
            kv_storage = self._dual_storage.get_kv_storage()
            kv_data_dict = await kv_storage.batch_get(keys=list(valid_event_ids))

            # Reconstruct full MemCells from KV-Storage
            result_dict = {}
            for event_id in valid_event_ids:
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
                "✅ Successfully batch retrieved MemCell by event_ids: requested %d, found %d, projection: %s",
                len(event_ids),
                len(result_dict),
                "yes" if projection_model else "no",
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
            # Note: MemCellLite inherits AuditBase, which will auto-set created_at/updated_at on insert
            memcell_lite = self._memcell_to_lite(memcell)
            await memcell_lite.insert(session=session)

            # Copy generated ID and audit fields back to full MemCell
            # (AuditBase has set these fields automatically during insert)
            memcell.id = memcell_lite.id
            memcell.created_at = memcell_lite.created_at
            memcell.updated_at = memcell_lite.updated_at

            logger.info(f"✅ MemCell appended to MongoDB: {memcell.event_id}")

            # 2. Write to KV-Storage (always full MemCell)
            success = await self._dual_storage.write_to_kv(memcell)
            return memcell if success else None
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

            # 3. Update MongoDB (only indexed fields in MemCellLite)
            # Find and update lite document
            lite_doc = await self.model.find_one({"_id": ObjectId(event_id)})
            if not lite_doc:
                logger.error(f"❌ MongoDB record not found for update: {event_id}")
                return None

            # Define indexed fields that can be updated
            indexed_fields = ['user_id', 'timestamp', 'group_id', 'participants', 'type', 'keywords']

            # Only update indexed fields that are present in update_data
            updated_count = 0
            for field in indexed_fields:
                if field in update_data:
                    setattr(lite_doc, field, getattr(memcell, field))
                    updated_count += 1

            # Save Lite model (AuditBase will automatically update updated_at)
            await lite_doc.save(session=session)
            logger.debug(f"✅ Updated {updated_count} indexed fields in MongoDB: {event_id}")

            # CRITICAL: Copy updated_at from Lite back to full MemCell for consistency
            # This ensures MongoDB and KV-Storage have the EXACT same timestamp
            memcell.updated_at = lite_doc.updated_at

            # 4. Update KV-Storage (always full MemCell)
            success = await self._dual_storage.write_to_kv(memcell)
            if not success:
                raise Exception(f"KV-Storage update failed for {event_id}")

            return memcell
        except Exception as e:
            logger.error("❌ Failed to update MemCell by event_id: %s", e)
            raise e

    async def delete_by_event_id(
        self,
        event_id: str,
        deleted_by: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Soft delete MemCell by event_id

        Soft delete only marks the record as deleted in MongoDB (sets deleted_at).
        KV-Storage data is NOT deleted during soft delete (needed for restore).

        Args:
            event_id: Event ID
            deleted_by: Deleter (optional)
            session: Optional MongoDB session, for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            # Soft delete: Only mark as deleted in MongoDB (set deleted_at field)
            # Query from MongoDB, not KV-Storage
            memcell_lite = await self.model.find_one({"_id": ObjectId(event_id)})
            if memcell_lite:
                await memcell_lite.delete(deleted_by=deleted_by, session=session)
                logger.debug(
                    "✅ Successfully soft deleted MemCell by event_id: %s", event_id
                )
                # NOTE: KV-Storage data is NOT deleted during soft delete
                # This allows restore operation to work properly
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to soft delete MemCell by event_id: %s", e)
            return False

    async def hard_delete_by_event_id(
        self, event_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Hard delete (physical deletion) MemCell by event_id from both MongoDB and KV-Storage

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Strategy: Delete from KV-Storage first (data source), then MongoDB (index).
        This ensures data is removed before index, preventing orphaned references.

        Args:
            event_id: Event ID
            session: Optional MongoDB session, for transaction support

        Returns:
            Returns True if deletion succeeds, otherwise False
        """
        try:
            # 1. First delete from KV-Storage (data source)
            kv_deleted = await self._dual_storage.delete_from_kv(event_id)

            # 2. Then hard delete from MongoDB (index) - use hard_find_one + hard_delete
            mongo_deleted = False
            memcell_lite = await self.model.hard_find_one({"_id": ObjectId(event_id)})
            if memcell_lite:
                await memcell_lite.hard_delete(session=session)
                mongo_deleted = True
                logger.debug("✅ MemCell hard deleted from MongoDB: %s", event_id)
            else:
                logger.warning(f"⚠️  MemCell not found in MongoDB: {event_id}")

            # Return True if deleted from both storages (strict consistency)
            return kv_deleted and mongo_deleted
        except Exception as e:
            logger.error("❌ Failed to hard delete MemCell by event_id: %s", e)
            return False

    # ==================== Query Methods ====================

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
            query = self.model.find({"user_id": user_id})

            # Sorting
            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            # Pagination
            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by user ID: %s, found %d records",
                user_id,
                len(results),
            )

            full_memcells = await self._memcell_lite_to_full(results)
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
    ) -> List[MemCell]:
        """
        Query MemCell by user ID and time range

        Check both user_id field and participants array, match if user_id is in either

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
            # Check both user_id field and participants array
            # Use OR logic: user_id matches OR user_id is in participants
            # Note: MongoDB automatically checks if array contains the value when using Eq on array fields
            query = self.model.find(
                And(
                    Or(
                        Eq(MemCellLite.user_id, user_id),
                        Eq(
                            MemCellLite.participants, user_id
                        ),  # MongoDB checks if array contains the value
                    ),
                    GTE(MemCellLite.timestamp, start_time),
                    LT(MemCellLite.timestamp, end_time),
                )
            ).sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by user and time range: %s, time range: %s - %s, found %d records",
                user_id,
                start_time,
                end_time,
                len(results),
            )

            full_memcells = await self._memcell_lite_to_full(results)
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
            query = self.model.find({"group_id": group_id})

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
                "✅ Successfully queried MemCell by group ID: %s, found %d records",
                group_id,
                len(results),
            )

            full_memcells = await self._memcell_lite_to_full(results)
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
        sort_desc: bool = False,
    ) -> List[MemCell]:
        """
        Query MemCell by time range

        Args:
            start_time: Start time
            end_time: End time
            limit: Limit number of returned results
            skip: Number of results to skip
            sort_desc: Whether to sort by time in descending order, default False (ascending)

        Returns:
            List of MemCell
        """
        try:
            query = self.model.find(
                {"timestamp": {"$gte": start_time, "$lt": end_time}}
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
                "✅ Successfully queried MemCell by time range: time range: %s - %s, found %d records",
                start_time,
                end_time,
                len(results),
            )

            full_memcells = await self._memcell_lite_to_full(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to query MemCell by time range: %s", e)
            import traceback

            logger.error("Detailed error information: %s", traceback.format_exc())
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
            if match_all:
                # Match all participants
                query = self.model.find({"participants": {"$all": participants}})
            else:
                # Match any participant
                query = self.model.find({"participants": {"$in": participants}})

            query = query.sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by participants: %s, match mode: %s, found %d records",
                participants,
                'all' if match_all else 'any',
                len(results),
            )

            full_memcells = await self._memcell_lite_to_full(results)
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
            if match_all:
                query = self.model.find({"keywords": {"$all": keywords}})
            else:
                query = self.model.find({"keywords": {"$in": keywords}})

            query = query.sort("-timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully queried MemCell by keywords: %s, match mode: %s, found %d records",
                keywords,
                'all' if match_all else 'any',
                len(results),
            )

            full_memcells = await self._memcell_lite_to_full(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to query MemCell by keywords: %s", e)
            return []

    # ==================== Batch Operations ====================

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
        Hard delete (physical deletion) all MemCell of a user from both MongoDB and KV-Storage

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Strategy: Delete from KV-Storage first (data source), then MongoDB (index).

        Args:
            user_id: User ID
            session: Optional MongoDB session

        Returns:
            Number of hard deleted records from MongoDB
        """
        try:
            # 1. Query all event_ids to be deleted (for KV-Storage batch delete)
            memcells = await self.model.find({"user_id": user_id}).to_list()
            event_ids = [str(mc.id) for mc in memcells]
            logger.debug(f"Found {len(event_ids)} MemCells to delete for user: {user_id}")

            # 2. First batch delete from KV-Storage (data source)
            if event_ids:
                kv_storage = self._dual_storage.get_kv_storage()
                try:
                    kv_deleted_count = await kv_storage.batch_delete(keys=event_ids)
                    logger.debug(f"✅ KV-Storage batch delete: {kv_deleted_count} records")
                except Exception as kv_error:
                    logger.warning(f"⚠️  KV-Storage batch delete failed: {kv_error}")

            # 3. Then hard delete from MongoDB (index)
            result = await self.model.hard_delete_many(
                {"user_id": user_id}, session=session
            )
            count = result.deleted_count if result else 0
            logger.info(
                "✅ Successfully hard deleted all MemCell of user: %s, deleted %d records",
                user_id,
                count,
            )
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
        Hard delete (physical deletion) MemCell within time range from both MongoDB and KV-Storage

        ⚠️ Warning: This operation is irreversible! Use with caution.

        Strategy: Delete from KV-Storage first (data source), then MongoDB (index).

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter
            session: Optional MongoDB session, for transaction support

        Returns:
            Number of hard deleted records from MongoDB
        """
        try:
            # Build filter
            filter_dict = {"timestamp": {"$gte": start_time, "$lt": end_time}}
            if user_id:
                filter_dict["user_id"] = user_id

            # 1. Query all event_ids to be deleted (for KV-Storage batch delete)
            memcells = await self.model.find(filter_dict).to_list()
            event_ids = [str(mc.id) for mc in memcells]
            logger.debug(f"Found {len(event_ids)} MemCells to delete in time range")

            # 2. First batch delete from KV-Storage (data source)
            if event_ids:
                kv_storage = self._dual_storage.get_kv_storage()
                try:
                    kv_deleted_count = await kv_storage.batch_delete(keys=event_ids)
                    logger.debug(f"✅ KV-Storage batch delete: {kv_deleted_count} records")
                except Exception as kv_error:
                    logger.warning(f"⚠️  KV-Storage batch delete failed: {kv_error}")

            # 3. Then hard delete from MongoDB (index)
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

    # ==================== Statistics and Aggregation Queries ====================

    async def count_by_user_id(self, user_id: str) -> int:
        """
        Count number of MemCell for a user

        Args:
            user_id: User ID

        Returns:
            Number of records
        """
        try:
            count = await self.model.find({"user_id": user_id}).count()
            logger.debug(
                "✅ Successfully counted user MemCell: %s, total %d records",
                user_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count user MemCell: %s", e)
            return 0

    async def count_by_time_range(
        self, start_time: datetime, end_time: datetime, user_id: Optional[str] = None
    ) -> int:
        """
        Count number of MemCell within time range

        Args:
            start_time: Start time
            end_time: End time
            user_id: Optional user ID filter

        Returns:
            Number of records
        """
        try:
            conditions = [
                GTE(MemCellLite.timestamp, start_time),
                LT(MemCellLite.timestamp, end_time),
            ]

            if user_id:
                conditions.append(Eq(MemCellLite.user_id, user_id))

            count = await self.model.find(And(*conditions)).count()
            logger.debug(
                "✅ Successfully counted MemCell within time range: %s - %s, user: %s, total %d records",
                start_time,
                end_time,
                user_id or 'all',
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count MemCell within time range: %s", e)
            return 0

    async def get_latest_by_user(self, user_id: str, limit: int = 10) -> List[MemCell]:
        """
        Get latest MemCell records for a user

        Args:
            user_id: User ID
            limit: Limit on number of returned records

        Returns:
            List of MemCell
        """
        try:
            results = (
                await self.model.find({"user_id": user_id})
                .sort("-timestamp")
                .limit(limit)
                .to_list()
            )
            logger.debug(
                "✅ Successfully retrieved latest user MemCell: %s, returned %d records",
                user_id,
                len(results),
            )

            full_memcells = await self._memcell_lite_to_full(results)
            return full_memcells
        except Exception as e:
            logger.error("❌ Failed to retrieve latest user MemCell: %s", e)
            return []


# Export
__all__ = ["MemCellRawRepository"]
