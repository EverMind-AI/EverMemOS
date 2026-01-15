#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for ForesightRecordRawRepository with KV-Storage

This test file comprehensively tests all CRUD methods in ForesightRecordRawRepository
with the dual MongoDB + KV-Storage pattern. Each test follows the pattern:
1. Create test data (save)
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Modified methods tested (6 total):
- save
- get_by_id
- get_by_parent_episode_id
- get_by_user_id
- delete_by_id
- delete_by_parent_episode_id
"""

import asyncio
import pytest
import pytest_asyncio
import uuid
from datetime import timedelta
from typing import List, TYPE_CHECKING

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

# Delay imports to avoid loading beanie at module level
# These will be imported inside fixtures/functions when needed
if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
        ForesightRecord,
        ForesightRecordProjection,
    )
    from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
        ForesightRecordRawRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface


# ==================== Test Fixtures ====================


@pytest_asyncio.fixture
async def repository():
    """Get ForesightRecordRawRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
        ForesightRecordRawRepository,
    )

    repo = get_bean_by_type(ForesightRecordRawRepository)
    yield repo
    # Cleanup is handled by individual tests


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface

    kv = get_bean_by_type(KVStorageInterface)
    yield kv


@pytest.fixture
def test_user_id():
    """Generate unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_parent_episode_id():
    """Generate unique test parent episode ID"""
    return f"test_episode_{uuid.uuid4().hex[:8]}"


# ==================== Test Helpers ====================


def create_test_foresight_record(
    user_id: str,
    parent_episode_id: str,
    content: str = "Test foresight content",
    group_id: str = None,
    participants: List[str] = None,
):
    """Helper function to create a test ForesightRecord with all fields"""
    from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
        ForesightRecord,
    )

    # Create complete ForesightRecord structure
    return ForesightRecord(
        # Core required fields
        user_id=user_id,
        parent_episode_id=parent_episode_id,
        content=content,
        # Optional fields - user/group info
        user_name=f"TestUser_{user_id[-8:]}",
        group_id=group_id or f"group_{user_id}",
        group_name=f"TestGroup_{user_id[-8:]}",
        # Optional fields - time range
        start_time="2024-01-01",
        end_time="2024-12-31",
        duration_days=365,
        # Optional fields - event info
        participants=participants or [user_id, "Participant1", "Participant2"],
        # Optional fields - vector
        vector=[0.1, 0.2, 0.3] * 128,  # 384-dim vector
        vector_model="text-embedding-3-small",
        # Optional fields - evidence and extension
        evidence="Test evidence supporting this foresight",
        extend={
            "test_flag": True,
            "test_id": uuid.uuid4().hex,
            "confidence": 0.9,
            "priority": "high",
        },
    )


def assert_foresight_record_equal(fr1, fr2, check_id: bool = True):
    """Assert two ForesightRecord objects are equal (comparing all fields)"""
    if check_id:
        assert str(fr1.id) == str(fr2.id), "IDs don't match"

    # Core required fields
    assert fr1.user_id == fr2.user_id, "user_id doesn't match"
    assert fr1.parent_episode_id == fr2.parent_episode_id, "parent_episode_id doesn't match"
    assert fr1.content == fr2.content, "content doesn't match"

    # Optional fields - user/group info
    assert fr1.user_name == fr2.user_name, "user_name doesn't match"
    assert fr1.group_id == fr2.group_id, "group_id doesn't match"
    assert fr1.group_name == fr2.group_name, "group_name doesn't match"

    # Optional fields - time range
    assert fr1.start_time == fr2.start_time, "start_time doesn't match"
    assert fr1.end_time == fr2.end_time, "end_time doesn't match"
    assert fr1.duration_days == fr2.duration_days, "duration_days doesn't match"

    # Optional fields - event info
    assert set(fr1.participants or []) == set(
        fr2.participants or []
    ), "participants don't match"

    # Optional fields - vector
    if fr1.vector or fr2.vector:
        assert (fr1.vector is not None) == (
            fr2.vector is not None
        ), "vector existence doesn't match"
        if fr1.vector and fr2.vector:
            assert len(fr1.vector) == len(fr2.vector), "vector length doesn't match"
    assert fr1.vector_model == fr2.vector_model, "vector_model doesn't match"

    # Optional fields - evidence and extension
    assert fr1.evidence == fr2.evidence, "evidence doesn't match"
    assert fr1.extend == fr2.extend, "extend doesn't match"


async def verify_kv_storage(repository, memory_id: str) -> bool:
    """Verify data exists in KV-Storage"""
    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    kv_storage = repository._get_kv_storage()
    if not kv_storage:
        logger.warning("KV-Storage not available")
        return False

    kv_json = await kv_storage.get(key=memory_id)
    return kv_json is not None


# ==================== Test Cases ====================


def get_logger():
    """Helper to get logger instance"""
    from core.observation.logger import get_logger as _get_logger

    return _get_logger(__name__)


class TestBasicCRUD:
    """Test basic CRUD operations: Create, Read, Delete"""

    async def test_01_save_and_get_by_id(self, repository, test_user_id, test_parent_episode_id):
        """
        Test: save + get_by_id
        Flow: Create a ForesightRecord -> Read it back -> Verify data matches
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: save + get_by_id")

        # 1. Create test ForesightRecord
        original = create_test_foresight_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            content="Test foresight for get_by_id",
        )

        # 2. Save to repository
        created = await repository.save(original)
        assert created is not None, "Failed to save ForesightRecord"
        assert created.id is not None, "Created ForesightRecord should have ID"

        memory_id = str(created.id)
        logger.info(f"✅ Created ForesightRecord with ID: {memory_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, memory_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # 4. Read back using get_by_id
        retrieved = await repository.get_by_id(memory_id)
        assert retrieved is not None, "Failed to retrieve ForesightRecord"
        logger.info(f"✅ Retrieved ForesightRecord by ID")

        # 5. Verify data matches
        assert_foresight_record_equal(created, retrieved, check_id=True)
        logger.info(f"✅ Data integrity verified")

        # Cleanup
        await repository.delete_by_id(memory_id)

    async def test_02_save_and_get_by_id_with_projection(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: save + get_by_id with ForesightRecordProjection
        Flow: Create -> Read with Projection -> Verify no vector data
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: save + get_by_id (Projection)")

        from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
            ForesightRecordProjection,
        )

        # 1. Create and save test ForesightRecord
        original = create_test_foresight_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            content="Test foresight for projection",
        )
        created = await repository.save(original)
        memory_id = str(created.id)
        logger.info(f"✅ Created ForesightRecord: {memory_id}")

        # 2. Read back using Projection
        retrieved = await repository.get_by_id(memory_id, model=ForesightRecordProjection)
        assert retrieved is not None, "Failed to retrieve with Projection"
        assert isinstance(
            retrieved, ForesightRecordProjection
        ), "Should return ForesightRecordProjection"

        # 3. Verify Projection doesn't have vector field
        assert not hasattr(retrieved, "vector"), "Projection should not have vector field"
        assert retrieved.content == original.content, "content should match"
        assert retrieved.vector_model == original.vector_model, "vector_model should match"
        logger.info(f"✅ Projection verified: no vector, other fields match")

        # Cleanup
        await repository.delete_by_id(memory_id)

    async def test_03_delete_by_id(self, repository, test_user_id, test_parent_episode_id):
        """
        Test: save + delete_by_id + get_by_id
        Flow: Create -> Delete -> Verify deletion (MongoDB + KV)
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_id")

        # 1. Create test ForesightRecord
        original = create_test_foresight_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            content="Test foresight to be deleted",
        )
        created = await repository.save(original)
        assert created is not None
        memory_id = str(created.id)
        logger.info(f"✅ Created ForesightRecord: {memory_id}")

        # 2. Verify it exists
        retrieved = await repository.get_by_id(memory_id)
        assert retrieved is not None, "ForesightRecord should exist before deletion"

        # 3. Delete the ForesightRecord
        deleted = await repository.delete_by_id(memory_id)
        assert deleted is True, "Deletion should return True"
        logger.info(f"✅ Deleted ForesightRecord: {memory_id}")

        # 4. Verify it no longer exists
        retrieved_after = await repository.get_by_id(memory_id)
        assert retrieved_after is None, "ForesightRecord should not exist after deletion"
        logger.info(f"✅ Verified deletion: ForesightRecord not found")

        # 5. Verify KV-Storage cleanup
        kv_exists = await verify_kv_storage(repository, memory_id)
        assert not kv_exists, "KV-Storage should be cleaned up"
        logger.info(f"✅ KV-Storage cleaned up")

    async def test_04_get_by_user_id(self, repository, test_user_id, test_parent_episode_id):
        """
        Test: save + get_by_user_id
        Flow: Create 3 ForesightRecords for user -> Query by user_id -> Verify results
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_user_id")

        # 1. Create 3 ForesightRecords for the same user
        created_list = []
        for i in range(3):
            original = create_test_foresight_record(
                user_id=test_user_id,
                parent_episode_id=f"{test_parent_episode_id}_{i}",
                content=f"Foresight {i+1} for user query",
            )
            created = await repository.save(original)
            created_list.append(created)

        logger.info(
            f"✅ Created {len(created_list)} ForesightRecords for user: {test_user_id}"
        )

        # 2. Query by user_id
        results = await repository.get_by_user_id(test_user_id)
        assert len(results) >= 3, f"Expected at least 3 results, got {len(results)}"
        logger.info(f"✅ Found {len(results)} ForesightRecords for user")

        # 3. Verify all created ForesightRecords are in results
        result_ids = {str(fr.id) for fr in results}
        for created in created_list:
            assert (
                str(created.id) in result_ids
            ), f"Created ForesightRecord {created.id} not in results"

        logger.info(f"✅ All created ForesightRecords found in query results")

        # Cleanup
        for created in created_list:
            await repository.delete_by_id(str(created.id))


class TestParentEpisodeOperations:
    """Test operations related to parent_episode_id"""

    async def test_05_get_by_parent_episode_id(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: save + get_by_parent_episode_id
        Flow: Create 3 ForesightRecords with same parent -> Query by parent_episode_id -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_parent_episode_id")

        # 1. Create 3 ForesightRecords with same parent_episode_id
        created_list = []
        for i in range(3):
            original = create_test_foresight_record(
                user_id=test_user_id,
                parent_episode_id=test_parent_episode_id,
                content=f"Foresight {i+1} for parent episode query",
            )
            created = await repository.save(original)
            created_list.append(created)

        logger.info(
            f"✅ Created {len(created_list)} ForesightRecords for parent: {test_parent_episode_id}"
        )

        # 2. Query by parent_episode_id
        results = await repository.get_by_parent_episode_id(test_parent_episode_id)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        logger.info(f"✅ Found {len(results)} ForesightRecords for parent episode")

        # 3. Verify all created ForesightRecords are in results
        result_ids = {str(fr.id) for fr in results}
        for created in created_list:
            assert (
                str(created.id) in result_ids
            ), f"Created ForesightRecord {created.id} not in results"

        logger.info(f"✅ All created ForesightRecords found in query results")

        # Cleanup
        for created in created_list:
            await repository.delete_by_id(str(created.id))

    async def test_06_delete_by_parent_episode_id(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: save + delete_by_parent_episode_id + get_by_parent_episode_id
        Flow: Create 3 ForesightRecords for parent -> Delete all by parent -> Verify deletion
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_parent_episode_id")

        # 1. Create 3 ForesightRecords for the same parent_episode_id
        created_list = []
        for i in range(3):
            fr = create_test_foresight_record(
                user_id=test_user_id,
                parent_episode_id=test_parent_episode_id,
                content=f"Foresight {i+1} to be deleted",
            )
            created = await repository.save(fr)
            created_list.append(created)

        logger.info(
            f"✅ Created 3 ForesightRecords for parent: {test_parent_episode_id}"
        )

        # 2. Verify count before deletion
        results_before = await repository.get_by_parent_episode_id(test_parent_episode_id)
        count_before = len(results_before)
        assert count_before >= 3, f"Expected at least 3 records, got {count_before}"

        # 3. Delete all by parent_episode_id
        deleted_count = await repository.delete_by_parent_episode_id(test_parent_episode_id)
        assert (
            deleted_count >= 3
        ), f"Expected to delete at least 3, deleted {deleted_count}"
        logger.info(f"✅ Deleted {deleted_count} ForesightRecords for parent episode")

        # 4. Verify count after deletion
        results_after = await repository.get_by_parent_episode_id(test_parent_episode_id)
        count_after = len(results_after)
        assert (
            count_after == 0
        ), f"Expected 0 records after deletion, got {count_after}"
        logger.info(f"✅ Verified deletion: count = 0")

        # 5. Verify individual ForesightRecords are gone
        for created in created_list:
            retrieved = await repository.get_by_id(str(created.id))
            assert retrieved is None, f"ForesightRecord {created.id} should be deleted"

        # 6. Verify KV-Storage cleanup
        for created in created_list:
            kv_exists = await verify_kv_storage(repository, str(created.id))
            assert not kv_exists, f"KV-Storage should be cleaned up for {created.id}"

        logger.info(f"✅ All KV-Storage entries cleaned up")


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_07_get_nonexistent_id(self, repository):
        """
        Test: get_by_id with non-existent ID
        Expected: Should return None
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_id (non-existent)")

        fake_id = "000000000000000000000000"
        result = await repository.get_by_id(fake_id)

        assert result is None, "Non-existent ID should return None"
        logger.info(f"✅ Non-existent ID handled correctly: returned None")

    async def test_08_delete_nonexistent_id(self, repository):
        """
        Test: delete_by_id with non-existent ID
        Expected: Should complete without error (return value depends on implementation)
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_id (non-existent)")

        fake_id = "000000000000000000000000"
        result = await repository.delete_by_id(fake_id)

        # Note: Return value depends on implementation
        # MongoDB won't find it, but KV-Storage might return False for delete operation
        assert isinstance(result, bool), "Should return a boolean"
        logger.info(f"✅ Non-existent ID deletion handled correctly: returned {result}")

    async def test_09_verify_audit_fields(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: Verify created_at and updated_at are set correctly
        Ensures audit fields are properly managed with Dual Storage
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Verify created_at and updated_at fields")

        # 1. Create and save ForesightRecord
        original = create_test_foresight_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            content="Test audit fields",
        )
        created = await repository.save(original)
        assert created is not None, "save should return ForesightRecord"

        # 2. Verify audit fields are set after save
        assert (
            created.created_at is not None
        ), "❌ BUG: created_at should not be None!"
        assert (
            created.updated_at is not None
        ), "❌ BUG: updated_at should not be None!"
        logger.info(
            f"✅ After save: created_at={created.created_at}, updated_at={created.updated_at}"
        )

        # 3. Retrieve from KV-Storage and verify persistence
        retrieved = await repository.get_by_id(str(created.id))
        assert retrieved is not None, "get_by_id should return ForesightRecord"
        assert (
            retrieved.created_at is not None
        ), "❌ BUG: created_at should persist in KV-Storage!"
        assert (
            retrieved.updated_at is not None
        ), "❌ BUG: updated_at should persist in KV-Storage!"
        logger.info(
            f"✅ After retrieve: created_at={retrieved.created_at}, updated_at={retrieved.updated_at}"
        )

        # 4. Verify created_at equals updated_at for newly created records
        time_diff = abs((retrieved.created_at - retrieved.updated_at).total_seconds())
        assert (
            time_diff < 1
        ), "created_at and updated_at should be nearly identical for new records"
        logger.info(f"✅ created_at ≈ updated_at (diff: {time_diff:.6f}s)")

        # Cleanup
        await repository.delete_by_id(str(created.id))
        logger.info("✅ Audit fields verification passed")


# ==================== Main Test Runner ====================


if __name__ == "__main__":
    """
    Run all tests with pytest

    Usage:
        pytest tests/test_foresight_record_crud_complete.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
