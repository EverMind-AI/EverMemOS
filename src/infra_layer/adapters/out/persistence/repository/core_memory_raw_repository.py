from typing import List, Optional, Dict, Any, Tuple, Union
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.core_memory import (
    CoreMemory,
)
from infra_layer.adapters.out.persistence.document.memory.core_memory_lite import (
    CoreMemoryLite,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_helper import (
    DualStorageHelper,
)

logger = get_logger(__name__)


@repository("core_memory_raw_repository", primary=True)
class CoreMemoryRawRepository(BaseRepository[CoreMemoryLite]):
    """
    Core memory raw data repository

    Provides CRUD operations and query functions for core memory.
    A single document contains data of two memory types: BaseMemory and Profile.
    (Preference-related fields have been merged into Profile)
    """

    def __init__(self):
        super().__init__(CoreMemoryLite)
        self._dual_storage = DualStorageHelper[CoreMemory, CoreMemoryLite](
            model_name="CoreMemory", full_model=CoreMemory
        )

    # ==================== Helper Methods ====================

    def _core_to_lite(self, core: CoreMemory) -> CoreMemoryLite:
        """
        Convert full CoreMemory to CoreMemoryLite (only indexed fields).

        Args:
            core: Full CoreMemory object

        Returns:
            CoreMemoryLite with only indexed fields

        Note:
            Audit fields (created_at/updated_at) are not copied here.
            They will be automatically set by AuditBase during insert/update operations.
        """
        return CoreMemoryLite(
            id=core.id,
            user_id=core.user_id,
            version=core.version,
            is_latest=core.is_latest,
        )

    async def _core_lite_to_full(
        self, results: List[CoreMemoryLite]
    ) -> List[CoreMemory]:
        """
        Reconstruct full CoreMemory objects from KV-Storage.

        Args:
            results: List of CoreMemoryLite from MongoDB query

        Returns:
            List of full CoreMemory objects from KV-Storage
        """
        return await self._dual_storage.reconstruct_batch(results)

    # ==================== Version Management Methods ====================

    async def ensure_latest(
        self, user_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Ensure the latest version flag is correct for the specified user

        Find the latest version by user_id, set its is_latest to True, and set others to False.
        This is an idempotent operation and can be safely called repeatedly.

        Args:
            user_id: User ID
            session: Optional MongoDB session for transaction support

        Returns:
            Whether the update was successful
        """
        try:
            # Query only the most recent record (optimize performance)
            latest_version = await self.model.find_one(
                {"user_id": user_id}, sort=[("version", -1)], session=session
            )

            if not latest_version:
                logger.debug("ℹ️  No core memory found to update: user_id=%s", user_id)
                return True

            # Bulk update: set is_latest to False for all old versions
            old_versions_lite = await self.model.find(
                {"user_id": user_id, "version": {"$ne": latest_version.version}},
                session=session,
            ).to_list()

            # Update MongoDB lite records
            await self.model.find(
                {"user_id": user_id, "version": {"$ne": latest_version.version}},
                session=session,
            ).update_many({"$set": {"is_latest": False}})

            # Sync old versions to KV-Storage
            kv_storage = self._dual_storage.get_kv_storage()
            for lite_doc in old_versions_lite:
                try:
                    kv_json = await kv_storage.get(key=str(lite_doc.id))
                    if kv_json:
                        full_core = CoreMemory.model_validate_json(kv_json)
                        full_core.is_latest = False
                        full_core.updated_at = lite_doc.updated_at
                        await self._dual_storage.write_to_kv(full_core)
                except Exception as kv_error:
                    logger.warning(
                        f"⚠️  Failed to sync is_latest to KV for version {lite_doc.version}: {kv_error}"
                    )

            # Update the latest version's is_latest to True
            if latest_version.is_latest != True:
                latest_version.is_latest = True
                await latest_version.save(session=session)
                logger.debug(
                    "✅ Set latest version flag: user_id=%s, version=%s",
                    user_id,
                    latest_version.version,
                )

                # Sync latest version to KV-Storage
                try:
                    kv_json = await kv_storage.get(key=str(latest_version.id))
                    if kv_json:
                        full_core = CoreMemory.model_validate_json(kv_json)
                        full_core.is_latest = True
                        full_core.updated_at = latest_version.updated_at
                        await self._dual_storage.write_to_kv(full_core)
                except Exception as kv_error:
                    logger.warning(
                        f"⚠️  Failed to sync latest version to KV: {kv_error}"
                    )

            return True
        except Exception as e:
            logger.error(
                "❌ Failed to ensure latest version flag: user_id=%s, error=%s",
                user_id,
                e,
            )
            return False

    # ==================== Basic CRUD Methods ====================

    async def get_by_user_id(
        self,
        user_id: str,
        version_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Union[Optional[CoreMemory], List[CoreMemory]]:
        """
        Get core memory by user ID

        Args:
            user_id: User ID
            version_range: Version range (start, end), inclusive interval [start, end].
                          If not provided or None, get the latest version (sorted by version descending)
                          If provided, return all versions within the range
            session: Optional MongoDB session for transaction support

        Returns:
            If version_range is None, return a single CoreMemory or None
            If version_range is not None, return List[CoreMemory]
        """
        try:
            query_filter = {"user_id": user_id}

            # Handle version range query
            if version_range:
                start_version, end_version = version_range
                version_filter = {}
                if start_version is not None:
                    version_filter["$gte"] = start_version
                if end_version is not None:
                    version_filter["$lte"] = end_version
                if version_filter:
                    query_filter["version"] = version_filter

            # If no version range is specified, get the latest version (single result)
            if version_range is None:
                result = await self.model.find_one(
                    query_filter,
                    sort=[
                        ("version", -1)
                    ],  # Sort by version descending to get the latest
                    session=session,
                )
                if result:
                    logger.debug(
                        "✅ Successfully retrieved core memory by user ID: %s, version=%s",
                        user_id,
                        result.version,
                    )
                    # Reconstruct from KV-Storage
                    full_cores = await self._core_lite_to_full([result])
                    return full_cores[0] if full_cores else None
                else:
                    logger.debug("ℹ️  Core memory not found: user_id=%s", user_id)
                    return None
            else:
                # If version range is specified, get all matching versions
                results = await self.model.find(
                    query_filter,
                    sort=[("version", -1)],
                    session=session,  # Sort by version descending
                ).to_list()
                logger.debug(
                    "✅ Successfully retrieved core memory versions by user ID: %s, version_range=%s, found %d records",
                    user_id,
                    version_range,
                    len(results),
                )
                # Reconstruct from KV-Storage
                full_cores = await self._core_lite_to_full(results)
                return full_cores
        except Exception as e:
            logger.error("❌ Failed to retrieve core memory by user ID: %s", e)
            return None if version_range is None else []

    async def update_by_user_id(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[CoreMemory]:
        """
        Update core memory by user ID

        Args:
            user_id: User ID
            update_data: Update data
            version: Optional version number; if specified, update the specific version, otherwise update the latest version
            session: Optional MongoDB session for transaction support

        Returns:
            Updated CoreMemory or None
        """
        try:
            # Find the document to update
            if version is not None:
                # Update specific version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "version": version}, session=session
                )
            else:
                # Update latest version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id}, sort=[("version", -1)], session=session
                )

            if not existing_doc:
                logger.warning(
                    "⚠️  Core memory not found for update: user_id=%s, version=%s",
                    user_id,
                    version,
                )
                return None

            # Get full object from KV-Storage
            kv_storage = self._dual_storage.get_kv_storage()
            kv_json = await kv_storage.get(key=str(existing_doc.id))
            if not kv_json:
                logger.error(
                    "❌ Core memory not found in KV-Storage: id=%s", existing_doc.id
                )
                return None

            full_core = CoreMemory.model_validate_json(kv_json)

            # Update full object
            for key, value in update_data.items():
                if hasattr(full_core, key):
                    setattr(full_core, key, value)

            # Update lite object (only indexed fields)
            if "version" in update_data and hasattr(existing_doc, "version"):
                existing_doc.version = update_data["version"]
            if "is_latest" in update_data and hasattr(existing_doc, "is_latest"):
                existing_doc.is_latest = update_data["is_latest"]

            # Save updated lite document
            await existing_doc.save(session=session)

            # Copy audit fields back to full object
            full_core.updated_at = existing_doc.updated_at

            logger.debug(
                "✅ Successfully updated core memory by user ID: user_id=%s, version=%s",
                user_id,
                existing_doc.version,
            )

            # Write to KV-Storage
            success = await self._dual_storage.write_to_kv(full_core)
            return full_core if success else None
        except Exception as e:
            logger.error("❌ Failed to update core memory by user ID: %s", e)
            return None

    async def delete_by_user_id(
        self,
        user_id: str,
        version: Optional[str] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> bool:
        """
        Delete core memory by user ID

        Args:
            user_id: User ID
            version: Optional version number; if specified, delete only that version, otherwise delete all versions
            session: Optional MongoDB session for transaction support

        Returns:
            Whether deletion was successful
        """
        try:
            query_filter = {"user_id": user_id}
            if version is not None:
                query_filter["version"] = version

            if version is not None:
                # Delete specific version
                # 1. Query all IDs to delete first
                lite_docs = await self.model.find(query_filter, session=session).to_list()
                doc_ids = [str(doc.id) for doc in lite_docs]

                # 2. Delete from KV-Storage first
                kv_deleted_count = 0
                if doc_ids:
                    kv_storage = self._dual_storage.get_kv_storage()
                    try:
                        kv_deleted_count = await kv_storage.batch_delete(keys=doc_ids)
                        logger.debug(
                            f"✅ KV-Storage deleted: {kv_deleted_count} records"
                        )
                    except Exception as kv_error:
                        logger.warning(f"⚠️  KV-Storage delete failed: {kv_error}")

                # 3. Delete from MongoDB
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ Successfully deleted core memory by user ID and version: user_id=%s, version=%s",
                        user_id,
                        version,
                    )
                    # After deletion, ensure the latest version flag is correct
                    await self.ensure_latest(user_id, session)
                else:
                    logger.warning(
                        "⚠️  Core memory not found for deletion: user_id=%s, version=%s",
                        user_id,
                        version,
                    )
            else:
                # Delete all versions
                # 1. Query all IDs to delete first
                lite_docs = await self.model.find(query_filter, session=session).to_list()
                doc_ids = [str(doc.id) for doc in lite_docs]

                # 2. Delete from KV-Storage first
                kv_deleted_count = 0
                if doc_ids:
                    kv_storage = self._dual_storage.get_kv_storage()
                    try:
                        kv_deleted_count = await kv_storage.batch_delete(keys=doc_ids)
                        logger.debug(
                            f"✅ KV-Storage deleted: {kv_deleted_count} records"
                        )
                    except Exception as kv_error:
                        logger.warning(f"⚠️  KV-Storage delete failed: {kv_error}")

                # 3. Delete from MongoDB
                result = await self.model.find(query_filter, session=session).delete()
                deleted_count = (
                    result.deleted_count if hasattr(result, 'deleted_count') else 0
                )
                success = deleted_count > 0

                if success:
                    logger.debug(
                        "✅ Successfully deleted all core memory by user ID: user_id=%s, deleted %d records",
                        user_id,
                        deleted_count,
                    )
                else:
                    logger.warning(
                        "⚠️  Core memory not found for deletion: user_id=%s", user_id
                    )

            return success
        except Exception as e:
            logger.error("❌ Failed to delete core memory by user ID: %s", e)
            return False

    async def upsert_by_user_id(
        self,
        user_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[CoreMemory]:
        """
        Update or insert core memory by user ID

        If update_data contains a version field:
        - If that version exists, update it
        - If that version does not exist, create a new version (version must be provided)
        If update_data does not contain a version field:
        - Get the latest version and update it; if it doesn't exist, raise an error (version must be provided when creating)

        Args:
            user_id: User ID
            update_data: Data to update (must contain version field when creating a new version)
            session: Optional MongoDB session for transaction support

        Returns:
            Updated or created core memory record
        """
        try:
            version = update_data.get("version")

            if version is not None:
                # If version is specified, find the specific version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id, "version": version}, session=session
                )
            else:
                # If version is not specified, find the latest version
                existing_doc = await self.model.find_one(
                    {"user_id": user_id}, sort=[("version", -1)], session=session
                )

            if existing_doc:
                # Update existing record
                # Get full object from KV-Storage
                kv_storage = self._dual_storage.get_kv_storage()
                kv_json = await kv_storage.get(key=str(existing_doc.id))
                if not kv_json:
                    logger.error(
                        "❌ Core memory not found in KV-Storage: id=%s", existing_doc.id
                    )
                    return None

                full_core = CoreMemory.model_validate_json(kv_json)

                # Update full object
                for key, value in update_data.items():
                    if hasattr(full_core, key):
                        setattr(full_core, key, value)

                # Update lite object (only indexed fields)
                if "version" in update_data and hasattr(existing_doc, "version"):
                    existing_doc.version = update_data["version"]
                if "is_latest" in update_data and hasattr(existing_doc, "is_latest"):
                    existing_doc.is_latest = update_data["is_latest"]

                # Save updated lite document
                await existing_doc.save(session=session)

                # Copy audit fields back to full object
                full_core.updated_at = existing_doc.updated_at

                logger.debug(
                    "✅ Successfully updated existing core memory: user_id=%s, version=%s",
                    user_id,
                    existing_doc.version,
                )

                # Write to KV-Storage
                await self._dual_storage.write_to_kv(full_core)

                # If version was updated, ensure latest flag is correct
                if version is not None:
                    await self.ensure_latest(user_id, session)

                return full_core
            else:
                # When creating a new record, version must be provided
                if version is None:
                    logger.error(
                        "❌ Version field must be provided when creating new core memory: user_id=%s",
                        user_id,
                    )
                    raise ValueError(
                        f"Version field must be provided when creating new core memory: user_id={user_id}"
                    )

                # Create new record (update_data should already contain user_id)
                new_core = CoreMemory(**update_data)

                # 1. Write CoreMemoryLite to MongoDB (indexed fields only)
                core_lite = self._core_to_lite(new_core)
                await core_lite.insert(session=session)

                # Copy generated ID and audit fields back to full CoreMemory
                new_core.id = core_lite.id
                new_core.created_at = core_lite.created_at
                new_core.updated_at = core_lite.updated_at

                logger.info(
                    "✅ Successfully created new core memory: user_id=%s, version=%s",
                    user_id,
                    new_core.version,
                )

                # 2. Write to KV-Storage (always full CoreMemory)
                success = await self._dual_storage.write_to_kv(new_core)

                # After creation, ensure latest version flag is correct
                await self.ensure_latest(user_id, session)

                return new_core if success else None
        except ValueError:
            # Re-raise ValueError, do not catch it in Exception
            raise
        except Exception as e:
            logger.error("❌ Failed to update or create core memory: %s", e)
            return None

    # ==================== Field Extraction Methods ====================

    def get_base(self, memory: CoreMemory) -> Dict[str, Any]:
        """
        Get basic information

        Args:
            memory: CoreMemory instance

        Returns:
            Dictionary of basic information
        """
        return {
            "user_name": memory.user_name,
            "gender": memory.gender,
            "position": memory.position,
            "supervisor_user_id": memory.supervisor_user_id,
            "team_members": memory.team_members,
            "okr": memory.okr,
            "base_location": memory.base_location,
            "hiredate": memory.hiredate,
            "age": memory.age,
            "department": memory.department,
        }

    def get_profile(self, memory: CoreMemory) -> Dict[str, Any]:
        """
        Get personal profile

        Args:
            memory: CoreMemory instance

        Returns:
            Dictionary of personal profile
        """
        return {
            "hard_skills": memory.hard_skills,
            "soft_skills": memory.soft_skills,
            "personality": memory.personality,
            "projects_participated": memory.projects_participated,
            "user_goal": memory.user_goal,
            "work_responsibility": memory.work_responsibility,
            "working_habit_preference": memory.working_habit_preference,
            "interests": getattr(memory, "interests", None),
            "tendency": memory.tendency,
        }

    async def find_by_user_ids(
        self,
        user_ids: List[str],
        only_latest: bool = True,
        session: Optional[AsyncClientSession] = None,
    ) -> List[CoreMemory]:
        """
        Batch retrieve core memory by list of user IDs

        Args:
            user_ids: List of user IDs
            only_latest: Whether to retrieve only the latest version, default is True. Use is_latest field to filter latest versions in batch queries
            session: Optional MongoDB session for transaction support

        Returns:
            List of CoreMemory
        """
        try:
            if not user_ids:
                return []

            query_filter = {"user_id": {"$in": user_ids}}

            # In batch queries, use is_latest field to filter latest versions
            if only_latest:
                query_filter["is_latest"] = True

            query = self.model.find(query_filter, session=session)

            results = await query.to_list()
            logger.debug(
                "✅ Successfully retrieved core memory by user ID list: %d user IDs, only_latest=%s, found %d records",
                len(user_ids),
                only_latest,
                len(results),
            )
            # Reconstruct from KV-Storage
            full_cores = await self._core_lite_to_full(results)
            return full_cores
        except Exception as e:
            logger.error("❌ Failed to retrieve core memory by user ID list: %s", e)
            return []


# Export
__all__ = ["CoreMemoryRawRepository"]
