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
    DualStorageMixin,  # 添加双存储支持 - 自动拦截 MongoDB 调用
    BaseRepository[ClusterState],
):
    """
    ClusterState native CRUD repository

    Provides ClusterStorage compatible interface:
    - save_cluster_state(group_id, state) -> bool
    - load_cluster_state(group_id) -> Optional[Dict]
    - get_cluster_assignments(group_id) -> Dict[str, str]
    - clear(group_id) -> bool
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
                # Merge state with existing document data
                existing_data = existing.model_dump(exclude={"id", "revision_id"})
                existing_data.update(state)
                existing_data["group_id"] = group_id  # Ensure group_id is set

                # Create new document with merged data, preserving ID
                doc_id = existing.id
                cluster_state = ClusterState(**existing_data)
                cluster_state.id = doc_id

                # Delete old and insert new to work around save() issue
                await existing.delete()
                # Use raw insert bypassing Beanie's ID generation
                await cluster_state.get_pymongo_collection().insert_one(
                    cluster_state.model_dump(by_alias=True, exclude={"revision_id"})
                )

                # Sync to KV
                from core.di import get_bean_by_type
                from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import KVStorageInterface
                import json
                kv_storage = get_bean_by_type(KVStorageInterface)
                # Use model_dump and manual JSON conversion to avoid ExpressionField issues
                data_dict = cluster_state.model_dump(mode="json", exclude={"revision_id"})
                await kv_storage.put(key=str(doc_id), value=json.dumps(data_dict))

                logger.debug(f"Updated cluster state: group_id={group_id}")
                return cluster_state
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
            # Get all documents first to delete from KV storage
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
