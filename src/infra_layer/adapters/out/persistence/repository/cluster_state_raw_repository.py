"""
ClusterState native CRUD repository

Cluster state data access layer based on Beanie ODM.
Provides ClusterStorage compatible interface (duck typing).
"""

from typing import Optional, Dict, Any
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository

from infra_layer.adapters.out.persistence.document.memory.cluster_state import (
    ClusterState,
)
from infra_layer.adapters.out.persistence.document.memory.cluster_state_lite import (
    ClusterStateLite,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_helper import (
    DualStorageHelper,
)

logger = get_logger(__name__)


@repository("cluster_state_raw_repository", primary=True)
class ClusterStateRawRepository(BaseRepository[ClusterStateLite]):
    """
    ClusterState native CRUD repository

    Provides ClusterStorage compatible interface:
    - save_cluster_state(group_id, state) -> bool
    - load_cluster_state(group_id) -> Optional[Dict]
    - get_cluster_assignments(group_id) -> Dict[str, str]
    - clear(group_id) -> bool
    """

    def __init__(self):
        super().__init__(ClusterStateLite)
        self._dual_storage = DualStorageHelper[ClusterState, ClusterStateLite](
            model_name="ClusterState", full_model=ClusterState
        )

    def _cluster_state_to_lite(self, cluster_state: ClusterState) -> ClusterStateLite:
        """
        Convert full ClusterState to ClusterStateLite (only indexed fields).

        Args:
            cluster_state: Full ClusterState object

        Returns:
            ClusterStateLite with only indexed fields

        Note:
            Audit fields (created_at/updated_at) are not copied here.
            They will be automatically set by AuditBase during insert/update operations.
        """
        return ClusterStateLite(
            id=cluster_state.id,
            group_id=cluster_state.group_id,
        )

    async def _cluster_state_lite_to_full(
        self, lite: ClusterStateLite
    ) -> Optional[ClusterState]:
        """
        Reconstruct full ClusterState object from KV-Storage.

        Args:
            lite: ClusterStateLite from MongoDB query

        Returns:
            Full ClusterState object from KV-Storage or None
        """
        return await self._dual_storage.reconstruct_single(lite)

    # ==================== ClusterStorage interface implementation ====================

    async def save_cluster_state(self, group_id: str, state: Dict[str, Any]) -> bool:
        result = await self.upsert_by_group_id(group_id, state)
        return result is not None

    async def load_cluster_state(self, group_id: str) -> Optional[Dict[str, Any]]:
        cluster_state = await self.get_by_group_id(group_id)
        if cluster_state is None:
            return None
        return cluster_state.model_dump(exclude={"id", "revision_id"})

    async def clear(self, group_id: Optional[str] = None) -> bool:
        if group_id is None:
            await self.delete_all()
        else:
            await self.delete_by_group_id(group_id)
        return True

    # ==================== Native CRUD methods ====================

    async def get_by_group_id(self, group_id: str) -> Optional[ClusterState]:
        try:
            # Query MongoDB Lite model first
            lite_result = await self.model.find_one({"group_id": group_id})
            if not lite_result:
                logger.debug(f"ℹ️  ClusterState not found: group_id={group_id}")
                return None

            # Reconstruct from KV-Storage
            full_cluster_state = await self._cluster_state_lite_to_full(lite_result)
            if full_cluster_state:
                logger.debug(
                    f"✅ Retrieved ClusterState by group_id successfully: {group_id}"
                )
            return full_cluster_state
        except Exception as e:
            logger.error(
                f"Failed to retrieve cluster state: group_id={group_id}, error={e}"
            )
            return None

    async def upsert_by_group_id(
        self, group_id: str, state: Dict[str, Any]
    ) -> Optional[ClusterState]:
        try:
            # Check if already exists
            existing_lite = await self.model.find_one({"group_id": group_id})

            # Prepare full ClusterState object
            state["group_id"] = group_id
            cluster_state = ClusterState(**state)

            if existing_lite:
                # Update: reuse existing ID and audit fields
                cluster_state.id = existing_lite.id
                cluster_state.created_at = existing_lite.created_at
                cluster_state.updated_at = existing_lite.updated_at

                # Update Lite in MongoDB
                existing_lite.group_id = group_id
                await existing_lite.save()

                # Copy updated audit fields back
                cluster_state.created_at = existing_lite.created_at
                cluster_state.updated_at = existing_lite.updated_at

                logger.debug(f"Updated cluster state: group_id={group_id}")
            else:
                # Insert new Lite
                cluster_state_lite = self._cluster_state_to_lite(cluster_state)
                await cluster_state_lite.insert()

                # Copy generated ID and audit fields back
                cluster_state.id = cluster_state_lite.id
                cluster_state.created_at = cluster_state_lite.created_at
                cluster_state.updated_at = cluster_state_lite.updated_at

                logger.info(f"Created cluster state: group_id={group_id}")

            # Write to KV-Storage (always full ClusterState)
            success = await self._dual_storage.write_to_kv(cluster_state)
            return cluster_state if success else None
        except Exception as e:
            logger.error(
                f"Failed to save cluster state: group_id={group_id}, error={e}"
            )
            return None

    async def get_cluster_assignments(self, group_id: str) -> Dict[str, str]:
        try:
            cluster_state = await self.get_by_group_id(group_id)
            if cluster_state is None:
                return {}
            return cluster_state.eventid_to_cluster or {}
        except Exception as e:
            logger.error(
                f"Failed to retrieve cluster assignments: group_id={group_id}, error={e}"
            )
            return {}

    async def delete_by_group_id(self, group_id: str) -> bool:
        try:
            # Find existing Lite
            cluster_state_lite = await self.model.find_one({"group_id": group_id})
            if cluster_state_lite:
                cluster_state_id = str(cluster_state_lite.id)

                # Delete from MongoDB
                await cluster_state_lite.delete()

                # Delete from KV-Storage
                await self._dual_storage.delete_from_kv(cluster_state_id)

                logger.info(f"Deleted cluster state: group_id={group_id}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete cluster state: group_id={group_id}, error={e}"
            )
            return False

    async def delete_all(self) -> int:
        try:
            # Get all IDs first for KV-Storage deletion
            all_lites = await self.model.find_all().to_list()
            cluster_state_ids = [str(lite.id) for lite in all_lites]

            # Delete from MongoDB
            result = await self.model.delete_all()
            count = result.deleted_count if result else 0

            # Delete from KV-Storage
            if count > 0 and cluster_state_ids:
                kv_storage = self._dual_storage.get_kv_storage()
                try:
                    await kv_storage.batch_delete(cluster_state_ids)
                except Exception as kv_error:
                    logger.error(f"⚠️  KV-Storage batch delete error: {kv_error}")

            logger.info(f"Deleted all cluster states: {count} items")
            return count
        except Exception as e:
            logger.error(f"Failed to delete all cluster states: {e}")
            return 0
