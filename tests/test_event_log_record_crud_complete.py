#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for EventLogRecordRawRepository with KV-Storage

This test file comprehensively tests all CRUD methods in EventLogRecordRawRepository
with the dual MongoDB + KV-Storage pattern. Each test follows the pattern:
1. Create test data (save)
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Modified methods tested (7 total):
- save
- get_by_id
- get_by_parent_episode_id
- get_by_user_id
- find_by_time_range
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
    from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
        EventLogRecord,
        EventLogRecordProjection,
    )
    from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
        EventLogRecordRawRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage import KVStorageInterface


# ==================== Test Fixtures ====================


@pytest_asyncio.fixture
async def repository():
    """Get EventLogRecordRawRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
        EventLogRecordRawRepository,
    )

    repo = get_bean_by_type(EventLogRecordRawRepository)
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


def create_test_event_log_record(
    user_id: str,
    parent_episode_id: str,
    atomic_fact: str = "Test atomic fact",
    group_id: str = None,
    participants: List[str] = None,
    timestamp_offset: timedelta = None,
):
    """Helper function to create a test EventLogRecord with all fields"""
    from common_utils.datetime_utils import get_now_with_timezone
    from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
        EventLogRecord,
    )

    now = get_now_with_timezone()
    if timestamp_offset:
        now = now + timestamp_offset

    # Create complete EventLogRecord structure
    return EventLogRecord(
        # Core required fields
        user_id=user_id,
        parent_episode_id=parent_episode_id,
        atomic_fact=atomic_fact,
        timestamp=now,
        # Optional fields - user/group info
        user_name=f"TestUser_{user_id[-8:]}",
        group_id=group_id or f"group_{user_id}",
        group_name=f"TestGroup_{user_id[-8:]}",
        # Optional fields - event info
        participants=participants or [user_id, "Participant1", "Participant2"],
        event_type="Conversation",
        # Optional fields - vector
        vector=[0.1, 0.2, 0.3] * 128,  # 384-dim vector
        vector_model="text-embedding-3-small",
        # Optional fields - extension
        extend={
            "test_flag": True,
            "test_id": uuid.uuid4().hex,
            "priority": "high",
            "location": "Test Location",
        },
    )


def assert_event_log_record_equal(log1, log2, check_id: bool = True):
    """Assert two EventLogRecord objects are equal (comparing all fields)"""
    if check_id:
        assert str(log1.id) == str(log2.id), "IDs don't match"

    # Core required fields
    assert log1.user_id == log2.user_id, "user_id doesn't match"
    assert log1.parent_episode_id == log2.parent_episode_id, "parent_episode_id doesn't match"
    assert log1.atomic_fact == log2.atomic_fact, "atomic_fact doesn't match"

    # Timestamps might have microsecond differences, allow small tolerance
    if log1.timestamp and log2.timestamp:
        time_diff = abs((log1.timestamp - log2.timestamp).total_seconds())
        assert time_diff < 1, f"timestamp difference too large: {time_diff}s"

    # Optional fields - user/group info
    assert log1.user_name == log2.user_name, "user_name doesn't match"
    assert log1.group_id == log2.group_id, "group_id doesn't match"
    assert log1.group_name == log2.group_name, "group_name doesn't match"

    # Optional fields - event info
    assert set(log1.participants or []) == set(
        log2.participants or []
    ), "participants don't match"
    assert log1.event_type == log2.event_type, "event_type doesn't match"

    # Optional fields - vector
    if log1.vector or log2.vector:
        assert (log1.vector is not None) == (
            log2.vector is not None
        ), "vector existence doesn't match"
        if log1.vector and log2.vector:
            assert len(log1.vector) == len(log2.vector), "vector length doesn't match"
    assert log1.vector_model == log2.vector_model, "vector_model doesn't match"

    # Optional fields - extension
    assert log1.extend == log2.extend, "extend doesn't match"


async def verify_kv_storage(repository, log_id: str) -> bool:
    """Verify data exists in KV-Storage"""
    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    kv_storage = repository._dual_storage.get_kv_storage()
    if not kv_storage:
        logger.warning("KV-Storage not available")
        return False

    kv_json = await kv_storage.get(key=log_id)
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
        Flow: Create an EventLogRecord -> Read it back -> Verify data matches
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: save + get_by_id")

        # 1. Create test EventLogRecord
        original = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            atomic_fact="Test atomic fact for get_by_id",
        )

        # 2. Save to repository
        created = await repository.save(original)
        assert created is not None, "Failed to save EventLogRecord"
        assert created.id is not None, "Created EventLogRecord should have ID"

        log_id = str(created.id)
        logger.info(f"✅ Created EventLogRecord with ID: {log_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, log_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # 4. Read back using get_by_id
        retrieved = await repository.get_by_id(log_id)
        assert retrieved is not None, "Failed to retrieve EventLogRecord"
        logger.info(f"✅ Retrieved EventLogRecord by ID")

        # 5. Verify data matches
        assert_event_log_record_equal(created, retrieved, check_id=True)
        logger.info(f"✅ Data integrity verified")

        # Cleanup
        await repository.delete_by_id(log_id)

    async def test_02_save_and_get_by_id_with_projection(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: save + get_by_id with EventLogRecordProjection
        Flow: Create -> Read with Projection -> Verify no vector data
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: save + get_by_id (Projection)")

        from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
            EventLogRecordProjection,
        )

        # 1. Create and save test EventLogRecord
        original = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            atomic_fact="Test atomic fact for projection",
        )
        created = await repository.save(original)
        log_id = str(created.id)
        logger.info(f"✅ Created EventLogRecord: {log_id}")

        # 2. Read back using Projection
        retrieved = await repository.get_by_id(log_id, model=EventLogRecordProjection)
        assert retrieved is not None, "Failed to retrieve with Projection"
        assert isinstance(
            retrieved, EventLogRecordProjection
        ), "Should return EventLogRecordProjection"

        # 3. Verify Projection doesn't have vector field
        assert not hasattr(retrieved, "vector"), "Projection should not have vector field"
        assert retrieved.atomic_fact == original.atomic_fact, "atomic_fact should match"
        assert retrieved.vector_model == original.vector_model, "vector_model should match"
        logger.info(f"✅ Projection verified: no vector, other fields match")

        # Cleanup
        await repository.delete_by_id(log_id)

    async def test_03_delete_by_id(self, repository, test_user_id, test_parent_episode_id):
        """
        Test: save + delete_by_id + get_by_id
        Flow: Create -> Delete -> Verify deletion (MongoDB + KV)
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_id")

        # 1. Create test EventLogRecord
        original = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            atomic_fact="Test atomic fact to be deleted",
        )
        created = await repository.save(original)
        assert created is not None
        log_id = str(created.id)
        logger.info(f"✅ Created EventLogRecord: {log_id}")

        # 2. Verify it exists
        retrieved = await repository.get_by_id(log_id)
        assert retrieved is not None, "EventLogRecord should exist before deletion"

        # 3. Delete the EventLogRecord
        deleted = await repository.delete_by_id(log_id)
        assert deleted is True, "Deletion should return True"
        logger.info(f"✅ Deleted EventLogRecord: {log_id}")

        # 4. Verify it no longer exists
        retrieved_after = await repository.get_by_id(log_id)
        assert retrieved_after is None, "EventLogRecord should not exist after deletion"
        logger.info(f"✅ Verified deletion: EventLogRecord not found")

        # 5. Verify KV-Storage cleanup
        kv_exists = await verify_kv_storage(repository, log_id)
        assert not kv_exists, "KV-Storage should be cleaned up"
        logger.info(f"✅ KV-Storage cleaned up")

    async def test_04_get_by_user_id(self, repository, test_user_id, test_parent_episode_id):
        """
        Test: save + get_by_user_id
        Flow: Create 3 EventLogRecords for user -> Query by user_id -> Verify results
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_user_id")

        # 1. Create 3 EventLogRecords for the same user
        created_list = []
        for i in range(3):
            original = create_test_event_log_record(
                user_id=test_user_id,
                parent_episode_id=f"{test_parent_episode_id}_{i}",
                atomic_fact=f"Atomic fact {i+1} for user query",
                timestamp_offset=timedelta(minutes=i),
            )
            created = await repository.save(original)
            created_list.append(created)

        logger.info(
            f"✅ Created {len(created_list)} EventLogRecords for user: {test_user_id}"
        )

        # 2. Query by user_id
        results = await repository.get_by_user_id(test_user_id)
        assert len(results) >= 3, f"Expected at least 3 results, got {len(results)}"
        logger.info(f"✅ Found {len(results)} EventLogRecords for user")

        # 3. Verify all created EventLogRecords are in results
        result_ids = {str(log.id) for log in results}
        for created in created_list:
            assert (
                str(created.id) in result_ids
            ), f"Created EventLogRecord {created.id} not in results"

        logger.info(f"✅ All created EventLogRecords found in query results")

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
        Flow: Create 3 EventLogRecords with same parent -> Query by parent_episode_id -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_parent_episode_id")

        # 1. Create 3 EventLogRecords with same parent_episode_id
        created_list = []
        for i in range(3):
            original = create_test_event_log_record(
                user_id=test_user_id,
                parent_episode_id=test_parent_episode_id,
                atomic_fact=f"Atomic fact {i+1} for parent episode query",
                timestamp_offset=timedelta(minutes=i),
            )
            created = await repository.save(original)
            created_list.append(created)

        logger.info(
            f"✅ Created {len(created_list)} EventLogRecords for parent: {test_parent_episode_id}"
        )

        # 2. Query by parent_episode_id
        results = await repository.get_by_parent_episode_id(test_parent_episode_id)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        logger.info(f"✅ Found {len(results)} EventLogRecords for parent episode")

        # 3. Verify all created EventLogRecords are in results
        result_ids = {str(log.id) for log in results}
        for created in created_list:
            assert (
                str(created.id) in result_ids
            ), f"Created EventLogRecord {created.id} not in results"

        logger.info(f"✅ All created EventLogRecords found in query results")

        # Cleanup
        for created in created_list:
            await repository.delete_by_id(str(created.id))

    async def test_06_delete_by_parent_episode_id(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: save + delete_by_parent_episode_id + get_by_parent_episode_id
        Flow: Create 3 EventLogRecords for parent -> Delete all by parent -> Verify deletion
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_parent_episode_id")

        # 1. Create 3 EventLogRecords for the same parent_episode_id
        created_list = []
        for i in range(3):
            log = create_test_event_log_record(
                user_id=test_user_id,
                parent_episode_id=test_parent_episode_id,
                atomic_fact=f"Atomic fact {i+1} to be deleted",
            )
            created = await repository.save(log)
            created_list.append(created)

        logger.info(
            f"✅ Created 3 EventLogRecords for parent: {test_parent_episode_id}"
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
        logger.info(f"✅ Deleted {deleted_count} EventLogRecords for parent episode")

        # 4. Verify count after deletion
        results_after = await repository.get_by_parent_episode_id(test_parent_episode_id)
        count_after = len(results_after)
        assert (
            count_after == 0
        ), f"Expected 0 records after deletion, got {count_after}"
        logger.info(f"✅ Verified deletion: count = 0")

        # 5. Verify individual EventLogRecords are gone
        for created in created_list:
            retrieved = await repository.get_by_id(str(created.id))
            assert retrieved is None, f"EventLogRecord {created.id} should be deleted"

        # 6. Verify KV-Storage cleanup
        for created in created_list:
            kv_exists = await verify_kv_storage(repository, str(created.id))
            assert not kv_exists, f"KV-Storage should be cleaned up for {created.id}"

        logger.info(f"✅ All KV-Storage entries cleaned up")


class TestQueryMethods:
    """Test query methods: find by filters and time ranges"""

    async def test_07_find_by_time_range(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: save + find_by_time_range
        Flow: Create EventLogRecords at different times -> Query by time range -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_time_range")

        from common_utils.datetime_utils import get_now_with_timezone

        now = get_now_with_timezone()

        # 1. Create EventLogRecords at different times
        # Before range
        log_before = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=f"{test_parent_episode_id}_before",
            atomic_fact="Before time range",
            timestamp_offset=timedelta(hours=-2),
        )
        created_before = await repository.save(log_before)

        # Inside range
        log_inside1 = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=f"{test_parent_episode_id}_inside1",
            atomic_fact="Inside time range 1",
            timestamp_offset=timedelta(minutes=0),
        )
        created_inside1 = await repository.save(log_inside1)

        log_inside2 = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=f"{test_parent_episode_id}_inside2",
            atomic_fact="Inside time range 2",
            timestamp_offset=timedelta(minutes=5),
        )
        created_inside2 = await repository.save(log_inside2)

        # After range
        log_after = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=f"{test_parent_episode_id}_after",
            atomic_fact="After time range",
            timestamp_offset=timedelta(hours=2),
        )
        created_after = await repository.save(log_after)

        logger.info(f"✅ Created 4 EventLogRecords at different times")

        # 2. Query with time range
        start_time = now - timedelta(minutes=10)
        end_time = now + timedelta(minutes=20)

        results = await repository.find_by_time_range(
            start_time=start_time,
            end_time=end_time,
            user_id=test_user_id,
        )

        # 3. Verify only inside-range EventLogRecords are returned
        result_ids = {str(log.id) for log in results}

        assert (
            str(created_inside1.id) in result_ids
        ), "Inside range 1 should be included"
        assert (
            str(created_inside2.id) in result_ids
        ), "Inside range 2 should be included"
        assert (
            str(created_before.id) not in result_ids
        ), "Before range should be excluded"
        assert (
            str(created_after.id) not in result_ids
        ), "After range should be excluded"

        logger.info(f"✅ Time range query verified: 2 inside, 2 outside")

        # Cleanup
        for created in [created_before, created_inside1, created_inside2, created_after]:
            await repository.delete_by_id(str(created.id))


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_08_get_nonexistent_id(self, repository):
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

    async def test_09_delete_nonexistent_id(self, repository):
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

    async def test_10_verify_audit_fields(
        self, repository, test_user_id, test_parent_episode_id
    ):
        """
        Test: Verify created_at and updated_at are set correctly
        Ensures audit fields are properly managed with Dual Storage
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Verify created_at and updated_at fields")

        # 1. Create and save EventLogRecord
        original = create_test_event_log_record(
            user_id=test_user_id,
            parent_episode_id=test_parent_episode_id,
            atomic_fact="Test audit fields",
        )
        created = await repository.save(original)
        assert created is not None, "save should return EventLogRecord"

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
        assert retrieved is not None, "get_by_id should return EventLogRecord"
        assert (
            retrieved.created_at is not None
        ), "❌ BUG: created_at should persist in KV-Storage!"
        assert (
            retrieved.updated_at is not None
        ), "❌ BUG: updated_at should persist in KV-Storage!"
        logger.info(
            f"✅ After retrieve: created_at={retrieved.created_at}, updated_at={retrieved.updated_at}"
        )

        # 4. Verify timezones are consistent
        assert (
            retrieved.created_at.tzinfo == retrieved.timestamp.tzinfo
        ), "created_at timezone should match timestamp"
        assert (
            retrieved.updated_at.tzinfo == retrieved.timestamp.tzinfo
        ), "updated_at timezone should match timestamp"
        logger.info(f"✅ Timezone consistency verified: {retrieved.timestamp.tzinfo}")

        # 5. Verify created_at equals updated_at for newly created records
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
        pytest tests/test_event_log_record_crud_complete.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
