#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ClusterStateRawRepository with DualStorageMixin

Verify that DualStorageMixin works correctly for ClusterState.
Repository code remains unchanged, all dual storage logic is handled transparently by Mixin.
"""

import pytest
import pytest_asyncio
import uuid
from typing import TYPE_CHECKING

from core.observation.logger import get_logger

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.cluster_state_raw_repository import (
        ClusterStateRawRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.cluster_state_raw_repository import (
        ClusterStateRawRepository,
    )
    return get_bean_by_type(ClusterStateRawRepository)


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )
    return get_bean_by_type(KVStorageInterface)


@pytest.fixture
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


def create_test_cluster_state(group_id: str):
    """Create test ClusterState"""
    from infra_layer.adapters.out.persistence.document.memory.cluster_state import (
        ClusterState,
    )

    return ClusterState(
        group_id=group_id,
        event_ids=["event1", "event2", "event3"],
        timestamps=[1000.0, 2000.0, 3000.0],
        cluster_ids=["cluster1", "cluster1", "cluster2"],
        eventid_to_cluster={
            "event1": "cluster1",
            "event2": "cluster1",
            "event3": "cluster2",
        },
        next_cluster_idx=3,
        cluster_centroids={
            "cluster1": [0.1, 0.2, 0.3],
            "cluster2": [0.4, 0.5, 0.6],
        },
        cluster_counts={
            "cluster1": 2,
            "cluster2": 1,
        },
        cluster_last_ts={
            "cluster1": 2000.0,
            "cluster2": 3000.0,
        },
    )


@pytest.mark.asyncio
class TestClusterStateDualStorage:
    """Test ClusterState dual storage functionality"""

    async def test_01_save_cluster_state_syncs_to_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: save_cluster_state() syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState save_cluster_state() syncs to KV-Storage")

        # Create test data
        state_dict = {
            "event_ids": ["event1", "event2"],
            "timestamps": [1000.0, 2000.0],
            "cluster_ids": ["cluster1", "cluster1"],
            "eventid_to_cluster": {"event1": "cluster1", "event2": "cluster1"},
            "next_cluster_idx": 2,
            "cluster_centroids": {"cluster1": [0.1, 0.2, 0.3]},
            "cluster_counts": {"cluster1": 2},
            "cluster_last_ts": {"cluster1": 2000.0},
        }

        success = await repository.save_cluster_state(test_group_id, state_dict)
        assert success, "save_cluster_state failed"
        logger.info(f"✅ Saved cluster state: {test_group_id}")

        # Get the document to retrieve its ID
        cluster_state = await repository.get_by_group_id(test_group_id)
        assert cluster_state is not None, "Failed to retrieve saved cluster state"
        doc_id = str(cluster_state.id)

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"

        # Verify full data in KV
        from infra_layer.adapters.out.persistence.document.memory.cluster_state import (
            ClusterState,
        )

        kv_doc = ClusterState.model_validate_json(kv_value)
        assert kv_doc.group_id == test_group_id
        assert kv_doc.event_ids == ["event1", "event2"]
        assert kv_doc.cluster_centroids == {"cluster1": [0.1, 0.2, 0.3]}
        logger.info("✅ Test passed: save_cluster_state() syncs to KV-Storage")

    async def test_02_get_by_group_id_reads_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: get_by_group_id() reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState get_by_group_id() reads from KV-Storage")

        # Create test data
        test_data = create_test_cluster_state(test_group_id)
        saved = await test_data.insert()
        assert saved is not None
        doc_id = str(test_data.id)
        logger.info(f"✅ Created: {doc_id}")

        # Get by group_id
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None, "get_by_group_id failed"
        assert retrieved.group_id == test_group_id
        assert retrieved.event_ids == ["event1", "event2", "event3"]
        assert retrieved.cluster_centroids is not None
        logger.info("✅ Test passed: get_by_group_id() reads from KV-Storage")

    async def test_03_load_cluster_state_works(
        self, repository, kv_storage, test_group_id
    ):
        """Test: load_cluster_state() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState load_cluster_state() works with dual storage")

        # Create test data
        state_dict = {
            "event_ids": ["event1", "event2", "event3"],
            "timestamps": [1000.0, 2000.0, 3000.0],
            "cluster_ids": ["cluster1", "cluster1", "cluster2"],
            "eventid_to_cluster": {
                "event1": "cluster1",
                "event2": "cluster1",
                "event3": "cluster2",
            },
            "next_cluster_idx": 3,
            "cluster_centroids": {
                "cluster1": [0.1, 0.2, 0.3],
                "cluster2": [0.4, 0.5, 0.6],
            },
            "cluster_counts": {"cluster1": 2, "cluster2": 1},
            "cluster_last_ts": {"cluster1": 2000.0, "cluster2": 3000.0},
        }

        success = await repository.save_cluster_state(test_group_id, state_dict)
        assert success

        # Load cluster state
        loaded_state = await repository.load_cluster_state(test_group_id)
        assert loaded_state is not None, "load_cluster_state failed"
        assert loaded_state["group_id"] == test_group_id
        assert loaded_state["event_ids"] == ["event1", "event2", "event3"]
        assert loaded_state["cluster_centroids"] == {
            "cluster1": [0.1, 0.2, 0.3],
            "cluster2": [0.4, 0.5, 0.6],
        }

        logger.info("✅ Test passed: load_cluster_state() returns full data")

    async def test_04_upsert_by_group_id_updates_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: upsert_by_group_id() updates KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState upsert_by_group_id() updates KV-Storage")

        # Create initial data
        initial_state = {
            "event_ids": ["event1"],
            "timestamps": [1000.0],
            "cluster_ids": ["cluster1"],
            "eventid_to_cluster": {"event1": "cluster1"},
            "next_cluster_idx": 2,
        }

        result = await repository.upsert_by_group_id(test_group_id, initial_state)
        assert result is not None
        doc_id = str(result.id)

        # Verify KV has initial data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        # Update with new data
        updated_state = {
            "event_ids": ["event1", "event2"],
            "timestamps": [1000.0, 2000.0],
            "cluster_ids": ["cluster1", "cluster1"],
            "eventid_to_cluster": {"event1": "cluster1", "event2": "cluster1"},
            "next_cluster_idx": 2,
        }

        result = await repository.upsert_by_group_id(test_group_id, updated_state)
        assert result is not None
        assert str(result.id) == doc_id  # Same document ID

        # Verify KV has updated data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        from infra_layer.adapters.out.persistence.document.memory.cluster_state import (
            ClusterState,
        )

        kv_doc = ClusterState.model_validate_json(kv_value)
        assert kv_doc.event_ids == ["event1", "event2"]
        assert kv_doc.eventid_to_cluster == {"event1": "cluster1", "event2": "cluster1"}

        logger.info("✅ Test passed: upsert_by_group_id() updates KV-Storage")

    async def test_05_delete_by_group_id_removes_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: delete_by_group_id() removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState delete_by_group_id() removes from KV-Storage")

        # Create test data
        test_data = create_test_cluster_state(test_group_id)
        await test_data.insert()
        doc_id = str(test_data.id)

        # Verify KV has data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        # Delete
        success = await repository.delete_by_group_id(test_group_id)
        assert success, "delete_by_group_id should return True"

        # Verify KV removed
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have the data after delete"

        logger.info("✅ Test passed: delete_by_group_id() removes from KV-Storage")

    async def test_06_get_cluster_assignments_works(
        self, repository, kv_storage, test_group_id
    ):
        """Test: get_cluster_assignments() returns correct mapping from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState get_cluster_assignments() works")

        # Create test data with specific assignments
        state_dict = {
            "event_ids": ["event1", "event2", "event3"],
            "eventid_to_cluster": {
                "event1": "cluster1",
                "event2": "cluster1",
                "event3": "cluster2",
            },
        }

        success = await repository.save_cluster_state(test_group_id, state_dict)
        assert success

        # Get cluster assignments
        assignments = await repository.get_cluster_assignments(test_group_id)
        assert assignments == {
            "event1": "cluster1",
            "event2": "cluster1",
            "event3": "cluster2",
        }

        logger.info("✅ Test passed: get_cluster_assignments() returns correct mapping")

    async def test_07_clear_with_group_id_removes_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: clear(group_id) removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState clear(group_id) removes from KV")

        # Create test data
        test_data = create_test_cluster_state(test_group_id)
        await test_data.insert()
        doc_id = str(test_data.id)

        # Verify KV has data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        # Clear specific group
        success = await repository.clear(test_group_id)
        assert success

        # Verify KV removed
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have the data after clear"

        logger.info("✅ Test passed: clear(group_id) removes from KV")

    async def test_08_clear_all_removes_all_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: clear() without group_id removes all from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ClusterState clear() removes all from KV")

        # Create multiple test data
        group_ids = [f"{test_group_id}_{i}" for i in range(3)]
        doc_ids = []

        for group_id in group_ids:
            test_data = create_test_cluster_state(group_id)
            await test_data.insert()
            doc_ids.append(str(test_data.id))

        # Verify all in KV
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None

        # Clear all
        success = await repository.clear()
        assert success

        # Verify all removed from KV
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is None, f"KV-Storage should not have {doc_id} after clear"

        logger.info("✅ Test passed: clear() removes all from KV")

    async def test_09_multiple_groups_independent(
        self, repository, kv_storage, test_group_id
    ):
        """Test: Multiple groups maintain independent cluster states"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Multiple groups maintain independent cluster states")

        # Create states for multiple groups
        group1_id = f"{test_group_id}_1"
        group2_id = f"{test_group_id}_2"

        state1 = {
            "event_ids": ["event1", "event2"],
            "eventid_to_cluster": {"event1": "cluster1", "event2": "cluster1"},
        }

        state2 = {
            "event_ids": ["event3", "event4"],
            "eventid_to_cluster": {"event3": "cluster2", "event4": "cluster2"},
        }

        success1 = await repository.save_cluster_state(group1_id, state1)
        success2 = await repository.save_cluster_state(group2_id, state2)
        assert success1 and success2

        # Verify each group has its own state
        loaded1 = await repository.load_cluster_state(group1_id)
        loaded2 = await repository.load_cluster_state(group2_id)

        assert loaded1["event_ids"] == ["event1", "event2"]
        assert loaded2["event_ids"] == ["event3", "event4"]
        assert loaded1["eventid_to_cluster"] == {"event1": "cluster1", "event2": "cluster1"}
        assert loaded2["eventid_to_cluster"] == {"event3": "cluster2", "event4": "cluster2"}

        logger.info("✅ Test passed: Multiple groups maintain independent states")

    async def test_10_empty_cluster_state_handling(
        self, repository, kv_storage, test_group_id
    ):
        """Test: Handling of empty/new cluster state"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Empty cluster state handling")

        # Try to load non-existent state
        loaded = await repository.load_cluster_state(test_group_id)
        assert loaded is None, "Non-existent state should return None"

        # Get assignments for non-existent state
        assignments = await repository.get_cluster_assignments(test_group_id)
        assert assignments == {}, "Non-existent state should return empty dict"

        logger.info("✅ Test passed: Empty cluster state handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
