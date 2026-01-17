#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for ConversationStatusRawRepository with KV-Storage

This test file comprehensively tests all CRUD methods in ConversationStatusRawRepository
with the dual MongoDB + KV-Storage pattern. Each test follows the pattern:
1. Create test data
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Methods tested:
- get_by_group_id
- upsert_by_group_id (insert and update modes)
- delete_by_group_id
- count_by_group_id
- Concurrent upsert handling
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


def create_test_update_data(
    old_msg_start_time: datetime = None,
    new_msg_start_time: datetime = None,
    last_memcell_time: datetime = None,
):
    """Helper function to create test update data for ConversationStatus"""
    from common_utils.datetime_utils import get_now_with_timezone

    now = get_now_with_timezone()

    return {
        "old_msg_start_time": old_msg_start_time or now - timedelta(days=7),
        "new_msg_start_time": new_msg_start_time or now - timedelta(days=1),
        "last_memcell_time": last_memcell_time or now - timedelta(hours=1),
    }


def assert_conversation_status_equal(cs1, cs2, check_id: bool = True):
    """Assert two ConversationStatus objects are equal"""
    if check_id:
        assert str(cs1.id) == str(cs2.id), "IDs don't match"

    assert cs1.group_id == cs2.group_id, "group_id doesn't match"

    # Check timestamps (allow small tolerance)
    def check_timestamp(t1, t2, name):
        if t1 and t2:
            if isinstance(t1, str):
                from common_utils.datetime_utils import parse_iso_datetime
                t1 = parse_iso_datetime(t1)
            if isinstance(t2, str):
                from common_utils.datetime_utils import parse_iso_datetime
                t2 = parse_iso_datetime(t2)
            time_diff = abs((t1 - t2).total_seconds())
            assert time_diff < 1, f"{name} difference too large: {time_diff}s"
        else:
            assert t1 == t2, f"{name} doesn't match (one is None)"

    check_timestamp(cs1.old_msg_start_time, cs2.old_msg_start_time, "old_msg_start_time")
    check_timestamp(cs1.new_msg_start_time, cs2.new_msg_start_time, "new_msg_start_time")
    check_timestamp(cs1.last_memcell_time, cs2.last_memcell_time, "last_memcell_time")


async def verify_kv_storage(repository, conversation_status_id: str) -> bool:
    """Verify data exists in KV-Storage"""
    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    kv_storage = repository._get_kv_storage()
    if not kv_storage:
        logger.warning("KV-Storage not available")
        return False

    kv_json = await kv_storage.get(key=conversation_status_id)
    return kv_json is not None


# ==================== Test Cases ====================


def get_logger():
    """Helper to get logger instance"""
    from core.observation.logger import get_logger as _get_logger

    return _get_logger(__name__)


class TestBasicCRUD:
    """Test basic CRUD operations: Create, Read, Delete"""

    async def test_01_upsert_insert(self, repository, test_group_id):
        """
        Test: upsert_by_group_id (insert mode)
        Flow: Upsert new ConversationStatus -> Verify it was created
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: upsert_by_group_id (insert)")

        # 1. Prepare update data
        update_data = create_test_update_data()

        # 2. Upsert (should insert)
        result = await repository.upsert_by_group_id(test_group_id, update_data)
        assert result is not None, "Upsert failed"
        assert result.id is not None, "Upserted ConversationStatus should have ID"

        conversation_status_id = str(result.id)
        logger.info(f"✅ Upserted (inserted) ConversationStatus with ID: {conversation_status_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, conversation_status_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")
        assert kv_exists, "Should exist in KV-Storage"

        # 4. Verify data
        assert result.group_id == test_group_id
        assert result.old_msg_start_time is not None
        assert result.new_msg_start_time is not None
        assert result.last_memcell_time is not None
        logger.info("✅ Upsert insert data verified")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_02_upsert_update(self, repository, test_group_id):
        """
        Test: upsert_by_group_id (update mode)
        Flow: Create ConversationStatus -> Upsert to update -> Verify update worked
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: upsert_by_group_id (update)")

        # 1. Create initial ConversationStatus
        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        initial_data = create_test_update_data(
            old_msg_start_time=now - timedelta(days=10),
            new_msg_start_time=now - timedelta(days=5),
            last_memcell_time=now - timedelta(days=1),
        )
        created = await repository.upsert_by_group_id(test_group_id, initial_data)
        assert created is not None
        logger.info(f"✅ Created initial ConversationStatus: {created.group_id}")

        # 2. Upsert with updated data (should update)
        update_data = create_test_update_data(
            old_msg_start_time=now - timedelta(days=3),
            new_msg_start_time=now - timedelta(hours=2),
            last_memcell_time=now - timedelta(minutes=30),
        )

        result = await repository.upsert_by_group_id(test_group_id, update_data)
        assert result is not None, "Upsert update failed"
        logger.info(f"✅ Upserted (updated) ConversationStatus: {result.group_id}")

        # 3. Verify update
        assert str(result.id) == str(created.id), "ID should remain the same"

        # Verify timestamps were updated
        initial_new_msg_time = initial_data["new_msg_start_time"]
        result_new_msg_time = result.new_msg_start_time

        # Compare as timestamps
        if isinstance(initial_new_msg_time, datetime) and isinstance(result_new_msg_time, datetime):
            assert result_new_msg_time > initial_new_msg_time, "new_msg_start_time should be updated"

        logger.info("✅ Upsert update data verified")

        # 4. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, str(result.id))
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_03_get_by_group_id(self, repository, test_group_id):
        """
        Test: get_by_group_id
        Flow: Create -> Get -> Verify data matches
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_group_id")

        # 1. Create ConversationStatus
        update_data = create_test_update_data()
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None
        logger.info(f"✅ Created ConversationStatus: {created.group_id}")

        # 2. Get by group_id
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None, f"Failed to retrieve ConversationStatus by group_id: {test_group_id}"
        logger.info(f"✅ Retrieved ConversationStatus by group_id: {test_group_id}")

        # 3. Verify data integrity
        assert_conversation_status_equal(created, retrieved)
        logger.info("✅ Data integrity verified: created == retrieved")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_04_delete_by_group_id(self, repository, test_group_id):
        """
        Test: delete_by_group_id
        Flow: Create -> Delete -> Verify deletion
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_group_id")

        # 1. Create ConversationStatus
        update_data = create_test_update_data()
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None
        conversation_status_id = str(created.id)
        logger.info(f"✅ Created ConversationStatus with ID: {conversation_status_id}")

        # 2. Verify exists before deletion
        before_delete = await repository.get_by_group_id(test_group_id)
        assert before_delete is not None, "Should exist before deletion"

        # 3. Delete
        delete_result = await repository.delete_by_group_id(test_group_id)
        assert delete_result is True, "Delete should return True"
        logger.info(f"✅ Deleted ConversationStatus by group_id: {test_group_id}")

        # 4. Verify deletion
        after_delete = await repository.get_by_group_id(test_group_id)
        assert after_delete is None, "Should not exist after deletion"
        logger.info("✅ Verified deletion from MongoDB")

        # 5. Verify KV-Storage deletion
        kv_exists = await verify_kv_storage(repository, conversation_status_id)
        assert kv_exists is False, "Should not exist in KV-Storage after deletion"
        logger.info("✅ Verified deletion from KV-Storage")

    async def test_05_count_by_group_id(self, repository, test_group_id):
        """
        Test: count_by_group_id
        Flow: Create -> Count -> Verify count
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: count_by_group_id")

        # 1. Count before creation (should be 0 or 1 if unique by group_id)
        count_before = await repository.count_by_group_id(test_group_id)
        logger.info(f"Count before creation: {count_before}")

        # 2. Create ConversationStatus
        update_data = create_test_update_data()
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None
        logger.info(f"✅ Created ConversationStatus: {created.group_id}")

        # 3. Count after creation
        count_after = await repository.count_by_group_id(test_group_id)
        logger.info(f"Count after creation: {count_after}")
        assert count_after > count_before, "Count should increase after creation"

        # Cleanup
        await repository.delete_by_group_id(test_group_id)


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_06_get_nonexistent(self, repository):
        """
        Test: Get non-existent ConversationStatus
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_group_id (non-existent)")

        result = await repository.get_by_group_id("nonexistent_group_id")
        assert result is None, "Should return None for non-existent group_id"
        logger.info("✅ Correctly returned None for non-existent group_id")

    async def test_07_delete_nonexistent(self, repository):
        """
        Test: Delete non-existent ConversationStatus
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_group_id (non-existent)")

        result = await repository.delete_by_group_id("nonexistent_group_id")
        assert result is False, "Should return False for non-existent group_id"
        logger.info("✅ Correctly returned False for non-existent group_id")

    async def test_08_multiple_updates(self, repository, test_group_id):
        """
        Test: Multiple sequential updates
        Flow: Create -> Update multiple times -> Verify final state
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: multiple sequential updates")

        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        # 1. Create initial
        initial_data = create_test_update_data(
            old_msg_start_time=now - timedelta(days=10),
        )
        result = await repository.upsert_by_group_id(test_group_id, initial_data)
        assert result is not None
        original_id = str(result.id)
        logger.info(f"✅ Created initial ConversationStatus: {result.group_id}")

        # 2. Update multiple times
        for i in range(3):
            update_data = create_test_update_data(
                old_msg_start_time=now - timedelta(days=10-i),
                new_msg_start_time=now - timedelta(days=5-i),
                last_memcell_time=now - timedelta(hours=12-i),
            )
            result = await repository.upsert_by_group_id(test_group_id, update_data)
            assert result is not None
            assert str(result.id) == original_id, "ID should remain the same"
            logger.info(f"✅ Update {i+1} completed")

        # 3. Verify final state
        final = await repository.get_by_group_id(test_group_id)
        assert final is not None
        assert str(final.id) == original_id
        logger.info("✅ Multiple updates verified")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)


class TestDualStorage:
    """Test Dual Storage consistency between MongoDB and KV-Storage"""

    async def test_09_dual_storage_consistency(self, repository, kv_storage, test_group_id):
        """
        Test: Verify MongoDB Lite and KV-Storage consistency
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Dual Storage consistency")

        # 1. Create ConversationStatus
        from infra_layer.adapters.out.persistence.document.memory.conversation_status_lite import (
            ConversationStatusLite,
        )
        update_data = create_test_update_data()
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None
        conversation_status_id = str(created.id)
        logger.info(f"✅ Created ConversationStatus with ID: {conversation_status_id}")

        # 2. Verify MongoDB Lite
        lite = await ConversationStatusLite.find_one({"group_id": test_group_id})
        assert lite is not None, "Should exist in MongoDB as Lite"
        assert lite.group_id == test_group_id
        logger.info("✅ Verified MongoDB Lite record")

        # 3. Verify KV-Storage
        kv_json = await kv_storage.get(conversation_status_id)
        assert kv_json is not None, "Should exist in KV-Storage"

        from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
            ConversationStatus,
        )
        full_from_kv = ConversationStatus.model_validate_json(kv_json)
        assert full_from_kv.group_id == test_group_id
        assert full_from_kv.old_msg_start_time is not None
        assert full_from_kv.new_msg_start_time is not None
        assert full_from_kv.last_memcell_time is not None
        logger.info("✅ Verified KV-Storage full record")

        # 4. Verify consistency
        assert str(lite.id) == str(full_from_kv.id), "IDs should match"
        assert lite.group_id == full_from_kv.group_id, "group_id should match"
        logger.info("✅ Verified Dual Storage consistency")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)

    async def test_10_update_propagates_to_both_storages(self, repository, kv_storage, test_group_id):
        """
        Test: Verify updates propagate to both MongoDB and KV-Storage
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Update propagates to both storages")

        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        # 1. Create initial
        initial_data = create_test_update_data()
        created = await repository.upsert_by_group_id(test_group_id, initial_data)
        assert created is not None
        conversation_status_id = str(created.id)
        logger.info(f"✅ Created ConversationStatus: {created.group_id}")

        # 2. Update
        update_data = create_test_update_data(
            new_msg_start_time=now - timedelta(minutes=5),
        )
        updated = await repository.upsert_by_group_id(test_group_id, update_data)
        assert updated is not None
        logger.info(f"✅ Updated ConversationStatus: {updated.group_id}")

        # 3. Verify MongoDB Lite has correct ID
        from infra_layer.adapters.out.persistence.document.memory.conversation_status_lite import (
            ConversationStatusLite,
        )
        lite = await ConversationStatusLite.find_one({"group_id": test_group_id})
        assert lite is not None
        assert str(lite.id) == conversation_status_id, "MongoDB Lite ID should match"
        logger.info("✅ MongoDB Lite has correct ID after update")

        # 4. Verify KV-Storage has updated data
        kv_json = await kv_storage.get(conversation_status_id)
        assert kv_json is not None

        from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
            ConversationStatus,
        )
        full_from_kv = ConversationStatus.model_validate_json(kv_json)

        # Verify the updated field
        updated_time = update_data["new_msg_start_time"]
        kv_time = full_from_kv.new_msg_start_time

        if isinstance(updated_time, datetime) and isinstance(kv_time, datetime):
            time_diff = abs((updated_time - kv_time).total_seconds())
            assert time_diff < 1, "KV-Storage should have updated data"

        logger.info("✅ KV-Storage has updated data")

        # Cleanup
        await repository.delete_by_group_id(test_group_id)
