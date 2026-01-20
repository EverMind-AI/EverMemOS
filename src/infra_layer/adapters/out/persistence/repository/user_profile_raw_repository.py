"""
UserProfile native CRUD repository

User profile data access layer based on Beanie ODM.
Provides ProfileStorage compatible interface (duck typing).
"""

from typing import Optional, Dict, Any, List
from beanie.operators import Or, Eq
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from core.oxm.constants import MAGIC_ALL

from infra_layer.adapters.out.persistence.document.memory.user_profile import (
    UserProfile,
)
from infra_layer.adapters.out.persistence.document.memory.user_profile_lite import (
    UserProfileLite,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_helper import (
    DualStorageHelper,
)

logger = get_logger(__name__)


@repository("user_profile_raw_repository", primary=True)
class UserProfileRawRepository(BaseRepository[UserProfileLite]):
    """
    UserProfile native CRUD repository

    Provides ProfileStorage compatible interfaces:
    - save_profile(user_id, profile, metadata) -> bool
    - get_profile(user_id) -> Optional[Any]
    - get_all_profiles() -> Dict[str, Any]
    - get_profile_history(user_id, limit) -> List[Dict]
    - clear() -> bool
    """

    def __init__(self):
        super().__init__(UserProfileLite)
        self._dual_storage = DualStorageHelper[UserProfile, UserProfileLite](
            model_name="UserProfile", full_model=UserProfile
        )

    def _user_profile_to_lite(self, user_profile: UserProfile) -> UserProfileLite:
        """
        Convert full UserProfile to UserProfileLite (only indexed fields).

        Args:
            user_profile: Full UserProfile object

        Returns:
            UserProfileLite with only indexed fields

        Note:
            Audit fields (created_at/updated_at) are not copied here.
            They will be automatically set by AuditBase during insert/update operations.
        """
        return UserProfileLite(
            id=user_profile.id,
            user_id=user_profile.user_id,
            group_id=user_profile.group_id,
        )

    async def _user_profile_lite_to_full(
        self, lite: UserProfileLite
    ) -> Optional[UserProfile]:
        """
        Reconstruct full UserProfile object from KV-Storage.

        Args:
            lite: UserProfileLite from MongoDB query

        Returns:
            Full UserProfile object from KV-Storage or None
        """
        return await self._dual_storage.reconstruct_single(lite)

    # ==================== ProfileStorage interface implementation ====================

    async def save_profile(
        self, user_id: str, profile: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        metadata = metadata or {}
        group_id = metadata.get("group_id", "default")

        profile_data = profile.to_dict() if hasattr(profile, 'to_dict') else profile
        result = await self.upsert(user_id, group_id, profile_data, metadata)
        return result is not None

    async def get_profile(
        self, user_id: str, group_id: str = "default"
    ) -> Optional[Any]:
        user_profile = await self.get_by_user_and_group(user_id, group_id)
        if user_profile is None:
            return None
        return user_profile.profile_data

    async def get_all_profiles(self, group_id: str = "default") -> Dict[str, Any]:
        user_profiles = await self.get_all_by_group(group_id)
        return {up.user_id: up.profile_data for up in user_profiles}

    async def get_profile_history(
        self, user_id: str, group_id: str = "default", limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        user_profile = await self.get_by_user_and_group(user_id, group_id)
        if user_profile is None:
            return []

        history = [
            {
                "version": user_profile.version,
                "profile": user_profile.profile_data,
                "confidence": user_profile.confidence,
                "updated_at": user_profile.updated_at,
                "cluster_id": user_profile.last_updated_cluster,
                "memcell_count": user_profile.memcell_count,
            }
        ]
        return history[:limit] if limit else history

    async def clear(self, group_id: Optional[str] = None) -> bool:
        if group_id is None:
            await self.delete_all()
        else:
            await self.delete_by_group(group_id)
        return True

    # ==================== Native CRUD methods ====================

    async def get_by_user_and_group(
        self, user_id: str, group_id: str
    ) -> Optional[UserProfile]:
        try:
            # Query MongoDB Lite model first
            lite_result = await self.model.find_one(
                {"user_id": user_id, "group_id": group_id}
            )
            if not lite_result:
                logger.debug(
                    f"ℹ️  UserProfile not found: user_id={user_id}, group_id={group_id}"
                )
                return None

            # Reconstruct from KV-Storage
            full_user_profile = await self._user_profile_lite_to_full(lite_result)
            if full_user_profile:
                logger.debug(
                    f"✅ Retrieved UserProfile successfully: user_id={user_id}, group_id={group_id}"
                )
            return full_user_profile
        except Exception as e:
            logger.error(
                f"Failed to retrieve user profile: user_id={user_id}, group_id={group_id}, error={e}"
            )
            return None

    async def get_all_by_group(self, group_id: str) -> List[UserProfile]:
        try:
            # Query MongoDB Lite models
            lite_results = await self.model.find({"group_id": group_id}).to_list()

            # Reconstruct full objects from KV-Storage
            full_profiles = []
            for lite in lite_results:
                full_profile = await self._user_profile_lite_to_full(lite)
                if full_profile:
                    full_profiles.append(full_profile)

            return full_profiles
        except Exception as e:
            logger.error(
                f"Failed to retrieve group user profiles: group_id={group_id}, error={e}"
            )
            return []

    async def get_all_by_user(self, user_id: str, limit: int = 40) -> List[UserProfile]:
        try:
            # Query MongoDB Lite models
            lite_results = (
                await self.model.find({"user_id": user_id})
                .sort([("version", -1)])
                .limit(limit)
                .to_list()
            )

            # Reconstruct full objects from KV-Storage
            full_profiles = []
            for lite in lite_results:
                full_profile = await self._user_profile_lite_to_full(lite)
                if full_profile:
                    full_profiles.append(full_profile)

            return full_profiles
        except Exception as e:
            logger.error(f"Failed to get user profile: user_id={user_id}, error={e}")
            return []

    async def find_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        limit: Optional[int] = None,
    ) -> List[UserProfile]:
        """
        Retrieve list of user profiles by filters (user_id and/or group_id)

        Args:
            user_id: User ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by user_id
                - None or "": Filter for null/empty values (records with user_id as None or "")
                - Other values: Exact match
            group_id: Group ID
                - Not provided or MAGIC_ALL ("__all__"): Don't filter by group_id
                - None or "": Filter for null/empty values (records with group_id as None or "")
                - Other values: Exact match
            limit: Limit number of returned results

        Returns:
            List of UserProfile
        """
        try:
            # Build query conditions
            conditions = []

            # Handle user_id filter
            if user_id != MAGIC_ALL:
                if user_id == "" or user_id is None:
                    # Explicitly filter for null or empty string
                    conditions.append(
                        Or(Eq(UserProfileLite.user_id, None), Eq(UserProfileLite.user_id, ""))
                    )
                else:
                    conditions.append(UserProfileLite.user_id == user_id)

            # Handle group_id filter
            if group_id != MAGIC_ALL:
                if group_id == "" or group_id is None:
                    # Explicitly filter for null or empty string
                    conditions.append(
                        Or(Eq(UserProfileLite.group_id, None), Eq(UserProfileLite.group_id, ""))
                    )
                else:
                    conditions.append(UserProfileLite.group_id == group_id)

            # Build query
            if conditions:
                # Combine conditions with AND
                query = self.model.find(*conditions)
            else:
                # No conditions - find all
                query = self.model.find()

            # Sort by version descending
            query = query.sort([("version", -1)])

            # Apply limit
            if limit:
                query = query.limit(limit)

            lite_results = await query.to_list()

            # Reconstruct full objects from KV-Storage
            full_profiles = []
            for lite in lite_results:
                full_profile = await self._user_profile_lite_to_full(lite)
                if full_profile:
                    full_profiles.append(full_profile)

            logger.debug(
                "✅ Retrieved user profiles successfully: user_id=%s, group_id=%s, found %d records",
                user_id,
                group_id,
                len(full_profiles),
            )
            return full_profiles
        except Exception as e:
            logger.error("❌ Failed to retrieve user profiles: %s", e)
            return []

    async def upsert(
        self,
        user_id: str,
        group_id: str,
        profile_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[UserProfile]:
        try:
            metadata = metadata or {}

            # Check if already exists
            existing_lite = await self.model.find_one(
                {"user_id": user_id, "group_id": group_id}
            )

            # Prepare full UserProfile object
            if existing_lite:
                # Update: reconstruct existing profile from KV-Storage
                existing = await self._user_profile_lite_to_full(existing_lite)
                if not existing:
                    logger.error(
                        f"Failed to reconstruct existing profile from KV-Storage: user_id={user_id}, group_id={group_id}"
                    )
                    return None

                # Update fields
                existing.profile_data = profile_data
                existing.version += 1
                existing.confidence = metadata.get("confidence", existing.confidence)

                if "cluster_id" in metadata:
                    cluster_id = metadata["cluster_id"]
                    if cluster_id not in existing.cluster_ids:
                        existing.cluster_ids.append(cluster_id)
                    existing.last_updated_cluster = cluster_id

                if "memcell_count" in metadata:
                    existing.memcell_count = metadata["memcell_count"]

                # Update Lite in MongoDB
                existing_lite.user_id = user_id
                existing_lite.group_id = group_id
                await existing_lite.save()

                # Copy updated audit fields back
                existing.created_at = existing_lite.created_at
                existing.updated_at = existing_lite.updated_at

                user_profile = existing
                logger.debug(
                    f"Updated user profile: user_id={user_id}, group_id={group_id}, version={user_profile.version}"
                )
            else:
                # Insert new UserProfile
                user_profile = UserProfile(
                    user_id=user_id,
                    group_id=group_id,
                    profile_data=profile_data,
                    scenario=metadata.get("scenario", "group_chat"),
                    confidence=metadata.get("confidence", 0.0),
                    version=1,
                    cluster_ids=(
                        [metadata["cluster_id"]] if "cluster_id" in metadata else []
                    ),
                    memcell_count=metadata.get("memcell_count", 0),
                    last_updated_cluster=metadata.get("cluster_id"),
                )

                # Insert new Lite
                user_profile_lite = self._user_profile_to_lite(user_profile)
                await user_profile_lite.insert()

                # Copy generated ID and audit fields back
                user_profile.id = user_profile_lite.id
                user_profile.created_at = user_profile_lite.created_at
                user_profile.updated_at = user_profile_lite.updated_at

                logger.info(
                    f"Created user profile: user_id={user_id}, group_id={group_id}"
                )

            # Write to KV-Storage (always full UserProfile)
            success = await self._dual_storage.write_to_kv(user_profile)
            return user_profile if success else None
        except Exception as e:
            logger.error(
                f"Failed to save user profile: user_id={user_id}, group_id={group_id}, error={e}"
            )
            return None

    async def delete_by_group(self, group_id: str) -> int:
        try:
            # Get all IDs first for KV-Storage deletion
            all_lites = await self.model.find({"group_id": group_id}).to_list()
            user_profile_ids = [str(lite.id) for lite in all_lites]

            # Delete from MongoDB
            result = await self.model.find({"group_id": group_id}).delete()
            count = result.deleted_count if result else 0

            # Delete from KV-Storage
            if count > 0 and user_profile_ids:
                kv_storage = self._dual_storage.get_kv_storage()
                try:
                    await kv_storage.batch_delete(user_profile_ids)
                except Exception as kv_error:
                    logger.error(f"⚠️  KV-Storage batch delete error: {kv_error}")

            logger.info(
                f"Deleted group user profiles: group_id={group_id}, count={count}"
            )
            return count
        except Exception as e:
            logger.error(
                f"Failed to delete group user profiles: group_id={group_id}, error={e}"
            )
            return 0

    async def delete_all(self) -> int:
        try:
            # Get all IDs first for KV-Storage deletion
            all_lites = await self.model.find_all().to_list()
            user_profile_ids = [str(lite.id) for lite in all_lites]

            # Delete from MongoDB
            result = await self.model.delete_all()
            count = result.deleted_count if result else 0

            # Delete from KV-Storage
            if count > 0 and user_profile_ids:
                kv_storage = self._dual_storage.get_kv_storage()
                try:
                    await kv_storage.batch_delete(user_profile_ids)
                except Exception as kv_error:
                    logger.error(f"⚠️  KV-Storage batch delete error: {kv_error}")

            logger.info(f"Deleted all user profiles: {count} items")
            return count
        except Exception as e:
            logger.error(f"Failed to delete all user profiles: {e}")
            return 0
