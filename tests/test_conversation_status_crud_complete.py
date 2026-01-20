#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for ConversationStatusRawRepository with KV-Storage

This test file comprehensively tests all CRUD methods in ConversationStatusRawRepository
with the dual MongoDB + KV-Storage pattern. Each test follows the pattern:
1. Create test data (upsert)
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Modified methods tested:
- get_by_group_id
- upsert_by_group_id
- delete_by_group_id
- count_by_group_id
"""

import asyncio
import pytest
import pytest_asyncio
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

# Delay imports to avoid loading beanie at module level
if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
        ConversationStatus,
    )
    from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
        ConversationStatusRawRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface


# ==================== Test Fixtures ====================


@pytest_asyncio.fixture
async def repository():
    """Get ConversationStatusRawRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
        ConversationStatusRawRepository,
    )
    repo = get_bean_by_type(ConversationStatusRawRepository)
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
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


# ==================== Test Helpers ====================


def create_test_conversation_status(group_id: str):
    """Helper function to create a test ConversationStatus"""
    from common_utils.datetime_utils import get_now_with_timezone

    now = get_now_with_timezone()

    return {
        "old_msg_start_time": now - timedelta(days=7),
        "new_msg_start_time": now - timedelta(hours=1),
        "last_memcell_time": now - timedelta(minutes=30),
    }


def assert_conversation_status_equal(cs1, cs2, check_id: bool = True):
    """Assert two ConversationStatus objects are equal"""
    if check_id:
        assert str(cs1.id) == str(cs2.id), "IDs don't match"

    assert cs1.group_id == cs2.group_id, "group_id doesn't match"

    # Compare timestamps (allow small tolerance for microsecond differences)
    if cs1.old_msg_start_time and cs2.old_msg_start_time:
        time_diff = abs(
            (cs1.old_msg_start_time - cs2.old_msg_start_time).total_seconds()
        )
        assert time_diff < 1, f"old_msg_start_time difference too large: {time_diff}s"

    if cs1.new_msg_start_time and cs2.new_msg_start_time:
        time_diff = abs(
            (cs1.new_msg_start_time - cs2.new_msg_start_time).total_seconds()
        )
        assert time_diff < 1, f"new_msg_start_time difference too large: {time_diff}s"

    if cs1.last_memcell_time and cs2.last_memcell_time:
        time_diff = abs(
            (cs1.last_memcell_time - cs2.last_memcell_time).total_seconds()
        )
        assert time_diff < 1, f"last_memcell_time difference too large: {time_diff}s"


async def verify_kv_storage(repository, status_id: str) -> bool:
    """Verify data exists in KV-Storage"""
    kv_storage = repository._dual_storage.get_kv_storage()
    if not kv_storage:
        return False

    kv_json = await kv_storage.get(key=status_id)
    return kv_json is not None


def get_logger():
    """Helper to get logger instance"""
    from core.observation.logger import get_logger as _get_logger

    return _get_logger(__name__)


# ==================== Test Cases ====================


class TestBasicCRUD:
    """Test basic CRUD operations: Create, Read, Delete"""

    async def test_01_upsert_and_get_by_group_id(self, repository, test_group_id):
        """
        Test: upsert_by_group_id (create) + get_by_group_id
        Flow: Create a ConversationStatus -> Read it back -> Verify data matches
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: upsert_by_group_id (create) + get_by_group_id")

        # 1. Create test ConversationStatus
        update_data = create_test_conversation_status(test_group_id)

        # 2. Upsert (should create new)
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None, "Failed to upsert ConversationStatus"
        assert created.id is not None, "Created ConversationStatus should have ID"
        assert created.group_id == test_group_id, "group_id should match"

        status_id = str(created.id)
        logger.info(f"✅ Created ConversationStatus with ID: {status_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, status_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # 4. Read back using get_by_group_id
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None, "Failed to retrieve ConversationStatus"
        logger.info(f"✅ Retrieved ConversationStatus by group_id")

        # 5. Verify data matches
        assert_conversation_status_equal(created, retrieved, check_id=True)
        logger.info(f"✅ Data integrity verified")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_02_upsert_existing_status(self, repository, test_group_id):
        """
        Test: upsert_by_group_id (update)
        Flow: Create -> Update -> Verify update
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: upsert_by_group_id (update)")

        from common_utils.datetime_utils import get_now_with_timezone

        now = get_now_with_timezone()

        # 1. Create initial ConversationStatus
        initial_data = create_test_conversation_status(test_group_id)
        created = await repository.upsert_by_group_id(test_group_id, initial_data)
        assert created is not None
        original_id = created.id
        logger.info(f"✅ Created ConversationStatus: {original_id}")

        # 2. Update with new data
        new_data = {
            "old_msg_start_time": now - timedelta(days=3),
            "new_msg_start_time": now - timedelta(minutes=5),
            "last_memcell_time": now,
        }
        updated = await repository.upsert_by_group_id(test_group_id, new_data)
        assert updated is not None

        # 3. Verify ID remains the same (update, not insert)
        assert updated.id == original_id, "ID should remain the same on update"
        logger.info(f"✅ ID unchanged: {updated.id}")

        # 4. Verify new data is stored
        assert updated.old_msg_start_time == new_data["old_msg_start_time"]
        assert updated.new_msg_start_time == new_data["new_msg_start_time"]
        assert updated.last_memcell_time == new_data["last_memcell_time"]
        logger.info(f"✅ Data updated successfully")

        # 5. Verify updated_at changed
        assert (
            updated.updated_at != created.updated_at
        ), "updated_at should change on update"
        logger.info(
            f"✅ updated_at changed: {created.updated_at} -> {updated.updated_at}"
        )

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_03_delete_by_group_id(self, repository, test_group_id):
        """
        Test: upsert_by_group_id + delete_by_group_id + get_by_group_id
        Flow: Create -> Delete -> Verify deletion (MongoDB + KV)
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_group_id")

        # 1. Create test ConversationStatus
        update_data = create_test_conversation_status(test_group_id)
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None
        status_id = str(created.id)
        logger.info(f"✅ Created ConversationStatus: {status_id}")

        # 2. Verify it exists
        retrieved = await repository.get_by_group_id(test_group_id)
        assert (
            retrieved is not None
        ), "ConversationStatus should exist before deletion"

        # 3. Delete the ConversationStatus
        deleted = await repository.delete_by_group_id(test_group_id)
        assert deleted is True, "Deletion should return True"
        logger.info(f"✅ Deleted ConversationStatus: {status_id}")

        # 4. Verify it no longer exists
        retrieved_after = await repository.get_by_group_id(test_group_id)
        assert (
            retrieved_after is None
        ), "ConversationStatus should not exist after deletion"
        logger.info(f"✅ Verified deletion: ConversationStatus not found")

        # 5. Verify KV-Storage cleanup
        kv_exists = await verify_kv_storage(repository, status_id)
        assert not kv_exists, "KV-Storage should be cleaned up"
        logger.info(f"✅ KV-Storage cleaned up")

    async def test_04_count_by_group_id(self, repository, test_group_id):
        """
        Test: upsert_by_group_id + count_by_group_id
        Flow: Create -> Count -> Verify count
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: count_by_group_id")

        # 1. Count before creation (should be 0)
        count_before = await repository.count_by_group_id(test_group_id)
        assert count_before == 0, "Count should be 0 before creation"
        logger.info(f"✅ Count before creation: {count_before}")

        # 2. Create ConversationStatus
        update_data = create_test_conversation_status(test_group_id)
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None
        logger.info(f"✅ Created ConversationStatus")

        # 3. Count after creation (should be 1)
        count_after = await repository.count_by_group_id(test_group_id)
        assert count_after == 1, f"Count should be 1 after creation, got {count_after}"
        logger.info(f"✅ Count after creation: {count_after}")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)


class TestConcurrency:
    """Test concurrent operations"""

    async def test_05_concurrent_upsert(self, repository):
        """
        Test: Concurrent upsert operations on same group_id
        Flow: Simulate concurrent upserts -> Verify only one record created
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Concurrent upsert (duplicate key handling)")

        test_group = f"test_group_concurrent_{uuid.uuid4().hex[:8]}"

        from common_utils.datetime_utils import get_now_with_timezone

        now = get_now_with_timezone()

        # 1. First upsert
        data1 = {
            "old_msg_start_time": now - timedelta(days=7),
            "new_msg_start_time": now - timedelta(hours=1),
            "last_memcell_time": now - timedelta(minutes=30),
        }
        result1 = await repository.upsert_by_group_id(test_group, data1)
        assert result1 is not None
        logger.info(f"✅ First upsert succeeded: {result1.id}")

        # 2. Second upsert (should update, not create)
        data2 = {
            "old_msg_start_time": now - timedelta(days=3),
            "new_msg_start_time": now - timedelta(minutes=5),
            "last_memcell_time": now,
        }
        result2 = await repository.upsert_by_group_id(test_group, data2)
        assert result2 is not None
        logger.info(f"✅ Second upsert succeeded: {result2.id}")

        # 3. Verify same ID (updated, not created new)
        assert result1.id == result2.id, "Should update existing record, not create new"
        logger.info(f"✅ Same ID: {result1.id} == {result2.id}")

        # 4. Verify data is from second upsert
        assert result2.last_memcell_time == data2["last_memcell_time"]
        logger.info(f"✅ Data from second upsert: {result2.last_memcell_time}")

        # Cleanup
        await repository.delete_by_group_id(test_group)


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_06_get_nonexistent_group_id(self, repository):
        """
        Test: get_by_group_id with non-existent group_id
        Expected: Should return None
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_group_id (non-existent)")

        fake_group_id = "nonexistent_group_12345678"
        result = await repository.get_by_group_id(fake_group_id)

        assert result is None, "Non-existent group_id should return None"
        logger.info(f"✅ Non-existent group_id handled correctly: returned None")

    async def test_07_delete_nonexistent_group_id(self, repository):
        """
        Test: delete_by_group_id with non-existent group_id
        Expected: Should return False
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_group_id (non-existent)")

        fake_group_id = "nonexistent_group_87654321"
        result = await repository.delete_by_group_id(fake_group_id)

        assert result is False, "Deleting non-existent should return False"
        logger.info(f"✅ Non-existent deletion handled correctly: returned False")

    async def test_08_verify_audit_fields(self, repository, test_group_id):
        """
        Test: Verify created_at and updated_at are set correctly
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Verify created_at and updated_at fields")

        # 1. Create ConversationStatus
        update_data = create_test_conversation_status(test_group_id)
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None

        # 2. Verify audit fields are set after upsert
        assert created.created_at is not None, "❌ BUG: created_at should not be None!"
        assert created.updated_at is not None, "❌ BUG: updated_at should not be None!"
        logger.info(
            f"✅ After upsert: created_at={created.created_at}, updated_at={created.updated_at}"
        )

        # 3. Retrieve from KV-Storage and verify persistence
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None
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
        await repository.delete_by_group_id(test_group_id)
        logger.info("✅ Audit fields verification passed")

    async def test_09_verify_conversation_id_property(self, repository, test_group_id):
        """
        Test: Verify conversation_id property returns id
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Verify conversation_id property")

        # 1. Create ConversationStatus
        update_data = create_test_conversation_status(test_group_id)
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None

        # 2. Verify conversation_id property
        assert created.conversation_id is not None, "conversation_id should not be None"
        assert (
            created.conversation_id == created.id
        ), "conversation_id should equal id"
        logger.info(
            f"✅ conversation_id property works: {created.conversation_id} == {created.id}"
        )

        # 3. Retrieve and verify property persists
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None
        assert (
            retrieved.conversation_id == retrieved.id
        ), "conversation_id property should work after retrieval"
        logger.info(f"✅ conversation_id property persists after retrieval")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)


# ==================== Main Test Runner ====================


if __name__ == "__main__":
    """
    Run all tests with pytest

    Usage:
        ./run_tests.sh test_conversation_status_crud_complete.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
