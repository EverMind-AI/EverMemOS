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
from infra_layer.adapters.out.persistence.repository.dual_storage_mixin import (
    DualStorageMixin,
)

logger = get_logger(__name__)


@repository("cluster_state_raw_repository", primary=True)
class ClusterStateRawRepository(
    DualStorageMixin,  # Add dual storage support - automatically intercepts MongoDB calls
    BaseRepository[ClusterState],
):
    """
    ClusterState native CRUD repository

    Provides ClusterStorage compatible interface:
    - save_cluster_state(group_id, state) -> bool
    - load_cluster_state(group_id) -> Optional[Dict]
    - get_cluster_assignments(group_id) -> Dict[str, str]
    - clear(group_id) -> bool

    Dual Storage: DualStorageMixin automatically intercepts all MongoDB operations
    """

    def __init__(self):
        super().__init__(ClusterState)

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
            return await self.model.find_one({"group_id": group_id})
        except Exception as e:
            logger.error(
                f"Failed to retrieve cluster state: group_id={group_id}, error={e}"
            )
            return None

    async def upsert_by_group_id(
        self, group_id: str, state: Dict[str, Any]
    ) -> Optional[ClusterState]:
        try:
            existing = await self.model.find_one({"group_id": group_id})

            if existing:
                for key, value in state.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                await existing.save()
                logger.debug(f"Updated cluster state: group_id={group_id}")
                return existing
            else:
                state["group_id"] = group_id
                cluster_state = ClusterState(**state)
                await cluster_state.insert()
                logger.info(f"Created cluster state: group_id={group_id}")
                return cluster_state
        except Exception as e:
            logger.error(
                f"Failed to save cluster state: group_id={group_id}, error={e}"
            )
            return None

    async def get_cluster_assignments(self, group_id: str) -> Dict[str, str]:
        try:
            cluster_state = await self.model.find_one({"group_id": group_id})
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
            cluster_state = await self.model.find_one({"group_id": group_id})
            if cluster_state:
                await cluster_state.delete()
                logger.info(f"Deleted cluster state: group_id={group_id}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete cluster state: group_id={group_id}, error={e}"
            )
            return False

    async def delete_all(self) -> int:
        try:
            # Get all documents first to ensure KV-Storage deletion via DualStorageMixin
            all_docs = await self.model.find({}).to_list()
            count = 0
            for doc in all_docs:
                try:
                    await doc.delete()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete cluster state {doc.id}: {e}")
            logger.info(f"Deleted all cluster states: {count} items")
            return count
        except Exception as e:
            logger.error(f"Failed to delete all cluster states: {e}")
            return 0
