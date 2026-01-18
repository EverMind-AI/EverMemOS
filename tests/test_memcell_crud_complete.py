#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CRUD Test for MemCellRawRepository with KV-Storage

This test file comprehensively tests all modified CRUD methods in MemCellRawRepository
comparing against commit 14491fe. Each test follows the pattern:
1. Create test data (append)
2. Read/Query test data
3. Verify data consistency between MongoDB and KV-Storage
4. Verify data integrity (inserted == retrieved)

Modified methods tested (17 total):
- get_by_event_id
- get_by_event_ids
- append_memcell
- update_by_event_id
- delete_by_event_id
- find_by_user_id
- find_by_user_and_time_range
- find_by_group_id
- find_by_time_range
- find_by_participants
- search_by_keywords
- delete_by_user_id
- delete_by_time_range
- count_by_user_id
- count_by_time_range
- get_latest_by_user
- get_user_activity_summary
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
    from infra_layer.adapters.out.persistence.document.memory.memcell import (
        DataTypeEnum,
        MemCell,
    )
    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
        MemCellRawRepository,
    )
    from infra_layer.adapters.out.persistence.kv_storage import MemCellKVStorage


# ==================== Test Fixtures ====================


@pytest_asyncio.fixture
async def repository():
    """Get MemCellRawRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
        MemCellRawRepository,
    )
    repo = get_bean_by_type(MemCellRawRepository)
    yield repo
    # Cleanup is handled by individual tests


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage import MemCellKVStorage
    kv = get_bean_by_type(MemCellKVStorage)
    yield kv


@pytest.fixture
def test_user_id():
    """Generate unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


# ==================== Test Helpers ====================


def create_test_memcell(
    user_id: str,
    summary: str = "Test memory",
    group_id: str = None,
    participants: List[str] = None,
    keywords: List[str] = None,
    data_type: str = "CONVERSATION",
    timestamp_offset: timedelta = None,
):
    """Helper function to create a test MemCell with all fields"""
    from common_utils.datetime_utils import get_now_with_timezone
    from infra_layer.adapters.out.persistence.document.memory.memcell import (
        DataTypeEnum,
        MemCell,
        Message,
        RawData,
    )

    now = get_now_with_timezone()
    if timestamp_offset:
        now = now + timestamp_offset

    # Convert string to DataTypeEnum
    type_enum = getattr(DataTypeEnum, data_type)

    # Create complete message structure
    message = Message(
        content="This is a test conversation message",
        files=["https://example.com/test_file.pdf", "https://example.com/test_image.jpg"],
        extend={
            "sender": user_id,
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "platform": "TestPlatform",
            "timestamp": now.isoformat(),
        },
    )

    # Create raw data structure
    raw_data = RawData(
        data_type=type_enum,
        messages=[message],
        meta={
            "chat_id": f"chat_{uuid.uuid4().hex[:8]}",
            "platform": "TestPlatform",
            "session_id": f"session_{uuid.uuid4().hex[:8]}",
        },
    )

    # Create MemCell with all fields
    return MemCell(
        # Core fields (required)
        user_id=user_id,
        timestamp=now,
        summary=summary,

        # Optional fields - data fields
        group_id=group_id or f"group_{user_id}",
        original_data=[raw_data.model_dump()],  # Convert RawData to dict
        participants=participants or [user_id, "Bot", "TestUser"],
        type=type_enum,

        # Optional fields - metadata
        subject=f"Subject: {summary}",
        keywords=keywords or ["test", "memory", "conversation"],
        linked_entities=[f"entity_{uuid.uuid4().hex[:8]}", f"project_{uuid.uuid4().hex[:8]}"],

        # Optional fields - possibly unused but included for completeness
        episode=f"Episode: Test scenario for {user_id}",
        foresight_memories=[
            {"prediction": "Future interaction expected", "confidence": 0.85},
            {"prediction": "Follow-up action needed", "confidence": 0.72},
        ],
        event_log={
            "event_type": "test_event",
            "action": "create_memcell",
            "timestamp": now.isoformat(),
            "metadata": {"test": True},
        },
        extend={
            "test_flag": True,
            "test_id": uuid.uuid4().hex,
            "custom_field_1": "custom_value_1",
            "custom_field_2": 42,
        },
    )


def assert_memcell_equal(mc1, mc2, check_id: bool = True):
    """Assert two MemCells are equal (comparing all fields)"""
    if check_id:
        assert str(mc1.id) == str(mc2.id), "IDs don't match"

    # Core fields
    assert mc1.user_id == mc2.user_id, "user_id doesn't match"
    assert mc1.summary == mc2.summary, "summary doesn't match"

    # Timestamps might have microsecond differences, allow small tolerance
    if mc1.timestamp and mc2.timestamp:
        time_diff = abs((mc1.timestamp - mc2.timestamp).total_seconds())
        assert time_diff < 1, f"timestamp difference too large: {time_diff}s"

    # Optional fields - data fields
    assert mc1.type == mc2.type, "type doesn't match"
    assert mc1.group_id == mc2.group_id, "group_id doesn't match"
    assert mc1.original_data == mc2.original_data, "original_data doesn't match"
    assert set(mc1.participants or []) == set(mc2.participants or []), "participants don't match"

    # Optional fields - metadata
    assert mc1.subject == mc2.subject, "subject doesn't match"
    assert set(mc1.keywords or []) == set(mc2.keywords or []), "keywords don't match"
    assert set(mc1.linked_entities or []) == set(mc2.linked_entities or []), "linked_entities don't match"

    # Optional fields - possibly unused
    assert mc1.episode == mc2.episode, "episode doesn't match"
    assert mc1.foresight_memories == mc2.foresight_memories, "foresight_memories don't match"
    assert mc1.event_log == mc2.event_log, "event_log doesn't match"
    assert mc1.extend == mc2.extend, "extend doesn't match"


async def verify_kv_storage(repository, event_id: str) -> bool:
    """Verify data exists in KV-Storage"""
    from core.observation.logger import get_logger
    logger = get_logger(__name__)

    kv_storage = repository._dual_storage.get_kv_storage()
    if not kv_storage:
        logger.warning("KV-Storage not available")
        return False

    kv_json = await kv_storage.get(key=event_id)
    return kv_json is not None


# ==================== Test Cases ====================


def get_logger():
    """Helper to get logger instance"""
    from core.observation.logger import get_logger as _get_logger
    return _get_logger(__name__)


class TestBasicCRUD:
    """Test basic CRUD operations: Create, Read, Update, Delete"""

    async def test_01_append_and_get_by_event_id(self, repository, test_user_id):
        """
        Test: append_memcell + get_by_event_id
        Flow: Create a MemCell -> Read it back -> Verify data matches
        """
        logger = get_logger()
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: append_memcell + get_by_event_id")

        # 1. Create test MemCell
        original = create_test_memcell(
            user_id=test_user_id,
            summary="Test memory for get_by_event_id",
        )

        # 2. Append to repository
        created = await repository.append_memcell(original)
        assert created is not None, "Failed to append MemCell"
        assert created.id is not None, "Created MemCell should have ID"

        event_id = str(created.id)
        logger.info(f"✅ Created MemCell with ID: {event_id}")

        # 3. Verify KV-Storage
        kv_exists = await verify_kv_storage(repository, event_id)
        logger.info(f"KV-Storage: {'✅ Exists' if kv_exists else '⚠️  Not found'}")

        # 4. Read back using get_by_event_id
        retrieved = await repository.get_by_event_id(event_id)
        assert retrieved is not None, "Failed to retrieve MemCell"
        logger.info(f"✅ Retrieved MemCell by event_id")

        # 5. Verify data matches
        assert_memcell_equal(created, retrieved, check_id=True)
        logger.info(f"✅ Data integrity verified")

        # Cleanup
        await repository.delete_by_event_id(event_id)

    async def test_02_append_and_get_by_event_ids(self, repository, test_user_id):
        """
        Test: append_memcell + get_by_event_ids (batch read)
        Flow: Create 3 MemCells -> Batch read -> Verify all data matches
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: append_memcell + get_by_event_ids")

        # 1. Create 3 test MemCells
        created_list = []
        for i in range(3):
            original = create_test_memcell(
                user_id=test_user_id,
                summary=f"Test memory {i+1} for batch get",
            )
            created = await repository.append_memcell(original)
            assert created is not None
            created_list.append(created)
            logger.info(f"✅ Created MemCell {i+1}: {created.id}")

        event_ids = [str(mc.id) for mc in created_list]

        # 2. Batch read using get_by_event_ids
        result_dict = await repository.get_by_event_ids(event_ids)
        assert len(result_dict) == 3, f"Expected 3 results, got {len(result_dict)}"
        logger.info(f"✅ Batch retrieved {len(result_dict)} MemCells")

        # 3. Verify all data matches
        for created in created_list:
            event_id = str(created.id)
            retrieved = result_dict.get(event_id)
            assert retrieved is not None, f"MemCell {event_id} not found in results"
            assert_memcell_equal(created, retrieved, check_id=True)

        logger.info(f"✅ All {len(created_list)} MemCells verified")

        # Cleanup
        for event_id in event_ids:
            await repository.delete_by_event_id(event_id)

    async def test_03_update_by_event_id(self, repository, test_user_id):
        """
        Test: append_memcell + update_by_event_id + get_by_event_id
        Flow: Create -> Update -> Read -> Verify updated data
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: update_by_event_id")

        # 1. Create test MemCell
        original = create_test_memcell(
            user_id=test_user_id,
            summary="Original summary",
            keywords=["original", "test"],
        )
        created = await repository.append_memcell(original)
        assert created is not None
        event_id = str(created.id)
        logger.info(f"✅ Created MemCell: {event_id}")

        # 2. Update the MemCell
        update_data = {
            "summary": "Updated summary",
            "keywords": ["updated", "test", "modified"],
        }
        updated = await repository.update_by_event_id(event_id, update_data)
        assert updated is not None, "Failed to update MemCell"
        logger.info(f"✅ Updated MemCell: {event_id}")

        # 3. Read back
        retrieved = await repository.get_by_event_id(event_id)
        assert retrieved is not None

        # 4. Verify updates
        assert retrieved.summary == "Updated summary", "Summary not updated"
        assert set(retrieved.keywords) == {"updated", "test", "modified"}, "Keywords not updated"
        assert retrieved.user_id == test_user_id, "user_id should not change"
        logger.info(f"✅ Update verified: summary and keywords changed")

        # 5. Verify KV-Storage has updated data
        kv_exists = await verify_kv_storage(repository, event_id)
        assert kv_exists, "Updated data should exist in KV-Storage"

        # Cleanup
        await repository.delete_by_event_id(event_id)

    async def test_04_delete_by_event_id(self, repository, test_user_id):
        """
        Test: append_memcell + delete_by_event_id + get_by_event_id
        Flow: Create -> Delete -> Verify deletion (MongoDB + KV)
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_event_id")

        # 1. Create test MemCell
        original = create_test_memcell(
            user_id=test_user_id,
            summary="Test memory to be deleted",
        )
        created = await repository.append_memcell(original)
        assert created is not None
        event_id = str(created.id)
        logger.info(f"✅ Created MemCell: {event_id}")

        # 2. Verify it exists
        retrieved = await repository.get_by_event_id(event_id)
        assert retrieved is not None, "MemCell should exist before deletion"

        # 3. Delete the MemCell
        deleted = await repository.delete_by_event_id(event_id)
        assert deleted is True, "Deletion should return True"
        logger.info(f"✅ Deleted MemCell: {event_id}")

        # 4. Verify it no longer exists
        retrieved_after = await repository.get_by_event_id(event_id)
        assert retrieved_after is None, "MemCell should not exist after deletion"
        logger.info(f"✅ Verified deletion: MemCell not found")

        # 5. Verify KV-Storage cleanup
        kv_exists = await verify_kv_storage(repository, event_id)
        assert not kv_exists, "KV-Storage should be cleaned up"
        logger.info(f"✅ KV-Storage cleaned up")


class TestQueryMethods:
    """Test query methods: find, search, filter"""

    async def test_05_find_by_user_id(self, repository, test_user_id):
        """
        Test: append_memcell + find_by_user_id
        Flow: Create 3 MemCells for user -> Query by user_id -> Verify results
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_user_id")

        # 1. Create 3 MemCells for the same user
        created_list = []
        for i in range(3):
            original = create_test_memcell(
                user_id=test_user_id,
                summary=f"Memory {i+1} for user query",
                timestamp_offset=timedelta(minutes=i),
            )
            created = await repository.append_memcell(original)
            created_list.append(created)

        logger.info(f"✅ Created {len(created_list)} MemCells for user: {test_user_id}")

        # 2. Query by user_id
        results = await repository.find_by_user_id(test_user_id)
        assert len(results) >= 3, f"Expected at least 3 results, got {len(results)}"
        logger.info(f"✅ Found {len(results)} MemCells for user")

        # 3. Verify all created MemCells are in results
        result_ids = {str(mc.id) for mc in results}
        for created in created_list:
            assert str(created.id) in result_ids, f"Created MemCell {created.id} not in results"

        logger.info(f"✅ All created MemCells found in query results")

        # Cleanup
        for created in created_list:
            await repository.delete_by_event_id(str(created.id))

    async def test_06_find_by_user_and_time_range(self, repository, test_user_id):
        """
        Test: append_memcell + find_by_user_and_time_range
        Flow: Create MemCells at different times -> Query by time range -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_user_and_time_range")

        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        # 1. Create MemCells at different times
        # Before range
        mc_before = create_test_memcell(
            user_id=test_user_id,
            summary="Before time range",
            timestamp_offset=timedelta(hours=-2),
        )
        created_before = await repository.append_memcell(mc_before)

        # Inside range
        mc_inside1 = create_test_memcell(
            user_id=test_user_id,
            summary="Inside time range 1",
            timestamp_offset=timedelta(minutes=0),
        )
        created_inside1 = await repository.append_memcell(mc_inside1)

        mc_inside2 = create_test_memcell(
            user_id=test_user_id,
            summary="Inside time range 2",
            timestamp_offset=timedelta(minutes=5),
        )
        created_inside2 = await repository.append_memcell(mc_inside2)

        # After range
        mc_after = create_test_memcell(
            user_id=test_user_id,
            summary="After time range",
            timestamp_offset=timedelta(hours=2),
        )
        created_after = await repository.append_memcell(mc_after)

        logger.info(f"✅ Created 4 MemCells at different times")

        # 2. Query with time range
        start_time = now - timedelta(minutes=10)
        end_time = now + timedelta(minutes=20)

        results = await repository.find_by_user_and_time_range(
            user_id=test_user_id,
            start_time=start_time,
            end_time=end_time,
        )

        # 3. Verify only inside-range MemCells are returned
        result_ids = {str(mc.id) for mc in results}

        assert str(created_inside1.id) in result_ids, "Inside range 1 should be included"
        assert str(created_inside2.id) in result_ids, "Inside range 2 should be included"
        assert str(created_before.id) not in result_ids, "Before range should be excluded"
        assert str(created_after.id) not in result_ids, "After range should be excluded"

        logger.info(f"✅ Time range query verified: 2 inside, 2 outside")

        # Cleanup
        for created in [created_before, created_inside1, created_inside2, created_after]:
            await repository.delete_by_event_id(str(created.id))

    async def test_07_find_by_group_id(self, repository, test_user_id, test_group_id):
        """
        Test: append_memcell + find_by_group_id
        Flow: Create MemCells in different groups -> Query by group -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_group_id")

        # 1. Create MemCells in target group
        group_mcs = []
        for i in range(3):
            mc = create_test_memcell(
                user_id=test_user_id,
                summary=f"Group memory {i+1}",
                group_id=test_group_id,
            )
            created = await repository.append_memcell(mc)
            group_mcs.append(created)

        # Create MemCell in other group
        other_group_id = f"other_{test_group_id}"
        mc_other = create_test_memcell(
            user_id=test_user_id,
            summary="Other group memory",
            group_id=other_group_id,
        )
        created_other = await repository.append_memcell(mc_other)

        logger.info(f"✅ Created 3 MemCells in target group, 1 in other group")

        # 2. Query by target group_id
        results = await repository.find_by_group_id(test_group_id)
        result_ids = {str(mc.id) for mc in results}

        # 3. Verify only target group MemCells are returned
        for mc in group_mcs:
            assert str(mc.id) in result_ids, f"Group MemCell {mc.id} should be included"

        assert str(created_other.id) not in result_ids, "Other group MemCell should be excluded"

        logger.info(f"✅ Group query verified: {len(group_mcs)} in target group")

        # Cleanup
        for mc in group_mcs + [created_other]:
            await repository.delete_by_event_id(str(mc.id))

    async def test_08_find_by_time_range(self, repository, test_user_id):
        """
        Test: append_memcell + find_by_time_range
        Flow: Create MemCells at different times -> Query by time only -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_time_range")

        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        # 1. Create MemCells at different times
        mc_before = create_test_memcell(
            user_id=test_user_id,
            summary="Before range",
            timestamp_offset=timedelta(hours=-3),
        )
        created_before = await repository.append_memcell(mc_before)

        mc_inside = create_test_memcell(
            user_id=test_user_id,
            summary="Inside range",
            timestamp_offset=timedelta(minutes=0),
        )
        created_inside = await repository.append_memcell(mc_inside)

        logger.info(f"✅ Created 2 MemCells at different times")

        # 2. Query with time range
        start_time = now - timedelta(hours=1)
        end_time = now + timedelta(hours=1)

        results = await repository.find_by_time_range(
            start_time=start_time,
            end_time=end_time,
        )

        result_ids = {str(mc.id) for mc in results}

        # 3. Verify
        assert str(created_inside.id) in result_ids, "Inside range should be included"
        assert str(created_before.id) not in result_ids, "Before range should be excluded"

        logger.info(f"✅ Time range query verified")

        # Cleanup
        for mc in [created_before, created_inside]:
            await repository.delete_by_event_id(str(mc.id))

    async def test_09_find_by_participants(self, repository, test_user_id):
        """
        Test: append_memcell + find_by_participants
        Flow: Create MemCells with different participants -> Query -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_participants")

        # 1. Create MemCells with different participants
        mc_alice_bob = create_test_memcell(
            user_id=test_user_id,
            summary="Chat with Alice and Bob",
            participants=["Alice", "Bob"],
        )
        created_alice_bob = await repository.append_memcell(mc_alice_bob)

        mc_alice_charlie = create_test_memcell(
            user_id=test_user_id,
            summary="Chat with Alice and Charlie",
            participants=["Alice", "Charlie"],
        )
        created_alice_charlie = await repository.append_memcell(mc_alice_charlie)

        mc_bob_only = create_test_memcell(
            user_id=test_user_id,
            summary="Chat with Bob only",
            participants=["Bob"],
        )
        created_bob_only = await repository.append_memcell(mc_bob_only)

        logger.info(f"✅ Created 3 MemCells with different participants")

        # 2. Query for any participant = Alice
        results_any = await repository.find_by_participants(
            participants=["Alice"],
            match_all=False,
        )
        result_ids_any = {str(mc.id) for mc in results_any}

        # Should include both Alice chats
        assert str(created_alice_bob.id) in result_ids_any
        assert str(created_alice_charlie.id) in result_ids_any
        assert str(created_bob_only.id) not in result_ids_any

        logger.info(f"✅ Query (match_any Alice) verified: 2 results")

        # 3. Query for all participants = [Alice, Bob]
        results_all = await repository.find_by_participants(
            participants=["Alice", "Bob"],
            match_all=True,
        )
        result_ids_all = {str(mc.id) for mc in results_all}

        # Should only include the one with both Alice and Bob
        assert str(created_alice_bob.id) in result_ids_all
        assert str(created_alice_charlie.id) not in result_ids_all

        logger.info(f"✅ Query (match_all [Alice, Bob]) verified: 1 result")

        # Cleanup
        for mc in [created_alice_bob, created_alice_charlie, created_bob_only]:
            await repository.delete_by_event_id(str(mc.id))

    async def test_10_search_by_keywords(self, repository, test_user_id):
        """
        Test: append_memcell + search_by_keywords
        Flow: Create MemCells with different keywords -> Search -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: search_by_keywords")

        # 1. Create MemCells with different keywords
        mc_python = create_test_memcell(
            user_id=test_user_id,
            summary="Python programming",
            keywords=["python", "programming", "code"],
        )
        created_python = await repository.append_memcell(mc_python)

        mc_java = create_test_memcell(
            user_id=test_user_id,
            summary="Java programming",
            keywords=["java", "programming", "enterprise"],
        )
        created_java = await repository.append_memcell(mc_java)

        mc_cooking = create_test_memcell(
            user_id=test_user_id,
            summary="Cooking recipe",
            keywords=["cooking", "recipe", "food"],
        )
        created_cooking = await repository.append_memcell(mc_cooking)

        logger.info(f"✅ Created 3 MemCells with different keywords")

        # 2. Search for keyword "programming" (match any)
        results = await repository.search_by_keywords(
            keywords=["programming"],
            match_all=False,
        )
        result_ids = {str(mc.id) for mc in results}

        # Should include both programming-related
        assert str(created_python.id) in result_ids
        assert str(created_java.id) in result_ids
        assert str(created_cooking.id) not in result_ids

        logger.info(f"✅ Keyword search verified: 2 results for 'programming'")

        # Cleanup
        for mc in [created_python, created_java, created_cooking]:
            await repository.delete_by_event_id(str(mc.id))


class TestBatchOperations:
    """Test batch operations: delete multiple records"""

    async def test_11_delete_by_user_id(self, repository):
        """
        Test: append_memcell + delete_by_user_id + count_by_user_id
        Flow: Create 3 MemCells for user -> Delete all by user -> Verify deletion
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_user_id")

        # Use unique user ID for this test
        test_user = f"test_user_delete_{uuid.uuid4().hex[:8]}"

        # 1. Create 3 MemCells for the user
        created_list = []
        for i in range(3):
            mc = create_test_memcell(
                user_id=test_user,
                summary=f"Memory {i+1} to be deleted",
            )
            created = await repository.append_memcell(mc)
            created_list.append(created)

        logger.info(f"✅ Created 3 MemCells for user: {test_user}")

        # 2. Verify count before deletion
        count_before = await repository.count_by_user_id(test_user)
        assert count_before >= 3, f"Expected at least 3 records, got {count_before}"

        # 3. Delete all by user_id
        deleted_count = await repository.delete_by_user_id(test_user)
        assert deleted_count >= 3, f"Expected to delete at least 3, deleted {deleted_count}"
        logger.info(f"✅ Deleted {deleted_count} MemCells for user")

        # 4. Verify count after deletion
        count_after = await repository.count_by_user_id(test_user)
        assert count_after == 0, f"Expected 0 records after deletion, got {count_after}"
        logger.info(f"✅ Verified deletion: count = 0")

        # 5. Verify individual MemCells are gone
        for created in created_list:
            retrieved = await repository.get_by_event_id(str(created.id))
            assert retrieved is None, f"MemCell {created.id} should be deleted"

    async def test_12_delete_by_time_range(self, repository, test_user_id):
        """
        Test: append_memcell + delete_by_time_range + count_by_time_range
        Flow: Create MemCells at different times -> Delete by time range -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_time_range")

        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        # 1. Create MemCells at different times
        mc_old = create_test_memcell(
            user_id=test_user_id,
            summary="Old memory",
            timestamp_offset=timedelta(hours=-3),
        )
        created_old = await repository.append_memcell(mc_old)

        mc_recent1 = create_test_memcell(
            user_id=test_user_id,
            summary="Recent memory 1",
            timestamp_offset=timedelta(minutes=0),
        )
        created_recent1 = await repository.append_memcell(mc_recent1)

        mc_recent2 = create_test_memcell(
            user_id=test_user_id,
            summary="Recent memory 2",
            timestamp_offset=timedelta(minutes=5),
        )
        created_recent2 = await repository.append_memcell(mc_recent2)

        logger.info(f"✅ Created 3 MemCells at different times")

        # 2. Delete recent ones by time range
        start_time = now - timedelta(minutes=10)
        end_time = now + timedelta(minutes=20)

        deleted_count = await repository.delete_by_time_range(
            start_time=start_time,
            end_time=end_time,
        )
        assert deleted_count >= 2, f"Expected to delete at least 2, deleted {deleted_count}"
        logger.info(f"✅ Deleted {deleted_count} MemCells in time range")

        # 3. Verify recent ones are deleted
        retrieved_recent1 = await repository.get_by_event_id(str(created_recent1.id))
        retrieved_recent2 = await repository.get_by_event_id(str(created_recent2.id))
        assert retrieved_recent1 is None, "Recent 1 should be deleted"
        assert retrieved_recent2 is None, "Recent 2 should be deleted"

        # 4. Verify old one still exists
        retrieved_old = await repository.get_by_event_id(str(created_old.id))
        assert retrieved_old is not None, "Old memory should still exist"
        logger.info(f"✅ Verified: recent deleted, old preserved")

        # Cleanup
        await repository.delete_by_event_id(str(created_old.id))


class TestStatisticsAndAggregation:
    """Test statistics and aggregation methods"""

    async def test_13_count_by_user_id(self, repository):
        """
        Test: append_memcell + count_by_user_id
        Flow: Create 3 MemCells -> Count -> Verify count
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: count_by_user_id")

        # Use unique user ID for this test
        test_user = f"test_user_count_{uuid.uuid4().hex[:8]}"

        # 1. Create 3 MemCells
        created_list = []
        for i in range(3):
            mc = create_test_memcell(
                user_id=test_user,
                summary=f"Memory {i+1} for counting",
            )
            created = await repository.append_memcell(mc)
            created_list.append(created)

        logger.info(f"✅ Created 3 MemCells for user: {test_user}")

        # 2. Count by user_id
        count = await repository.count_by_user_id(test_user)
        assert count == 3, f"Expected count = 3, got {count}"
        logger.info(f"✅ Count verified: {count}")

        # Cleanup
        await repository.delete_by_user_id(test_user)

    async def test_14_count_by_time_range(self, repository, test_user_id):
        """
        Test: append_memcell + count_by_time_range
        Flow: Create MemCells at different times -> Count by range -> Verify
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: count_by_time_range")

        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        # 1. Create MemCells at different times
        created_list = []

        # Before range
        mc_before = create_test_memcell(
            user_id=test_user_id,
            summary="Before range",
            timestamp_offset=timedelta(hours=-2),
        )
        created_before = await repository.append_memcell(mc_before)
        created_list.append(created_before)

        # Inside range (2 MemCells)
        for i in range(2):
            mc = create_test_memcell(
                user_id=test_user_id,
                summary=f"Inside range {i+1}",
                timestamp_offset=timedelta(minutes=i*5),
            )
            created = await repository.append_memcell(mc)
            created_list.append(created)

        logger.info(f"✅ Created 3 MemCells at different times")

        # 2. Count in time range
        start_time = now - timedelta(minutes=10)
        end_time = now + timedelta(minutes=20)

        count = await repository.count_by_time_range(
            start_time=start_time,
            end_time=end_time,
        )
        assert count >= 2, f"Expected at least 2 in range, got {count}"
        logger.info(f"✅ Count in time range: {count}")

        # Cleanup
        for created in created_list:
            await repository.delete_by_event_id(str(created.id))

    async def test_15_get_latest_by_user(self, repository, test_user_id):
        """
        Test: append_memcell + get_latest_by_user
        Flow: Create 5 MemCells at different times -> Get latest 3 -> Verify order
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_latest_by_user")

        # 1. Create 5 MemCells at different times
        created_list = []
        for i in range(5):
            mc = create_test_memcell(
                user_id=test_user_id,
                summary=f"Memory {i+1}",
                timestamp_offset=timedelta(minutes=i),
            )
            created = await repository.append_memcell(mc)
            created_list.append(created)

        logger.info(f"✅ Created 5 MemCells for user")

        # 2. Get latest 3
        latest = await repository.get_latest_by_user(test_user_id, limit=3)
        assert len(latest) >= 3, f"Expected at least 3 results, got {len(latest)}"
        logger.info(f"✅ Retrieved {len(latest)} latest MemCells")

        # 3. Verify order (descending by timestamp)
        for i in range(len(latest) - 1):
            assert latest[i].timestamp >= latest[i+1].timestamp, \
                "Results should be ordered by timestamp descending"

        logger.info(f"✅ Order verified: descending by timestamp")

        # Cleanup
        for created in created_list:
            await repository.delete_by_event_id(str(created.id))

    async def test_16_get_user_activity_summary(self, repository, test_user_id):
        """
        Test: append_memcell + get_user_activity_summary
        Flow: Create multiple MemCells -> Get summary -> Verify stats
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_user_activity_summary")

        from common_utils.datetime_utils import get_now_with_timezone
        now = get_now_with_timezone()

        # 1. Create multiple MemCells
        created_list = []

        # Create 3 CONVERSATION type MemCells
        for i in range(3):
            mc = create_test_memcell(
                user_id=test_user_id,
                summary=f"Conversation {i+1}",
                data_type="CONVERSATION",
                timestamp_offset=timedelta(minutes=i),
            )
            created = await repository.append_memcell(mc)
            created_list.append(created)

        logger.info(f"✅ Created 3 MemCells (CONVERSATION type)")

        # 2. Get activity summary
        start_time = now - timedelta(minutes=10)
        end_time = now + timedelta(hours=1)

        summary = await repository.get_user_activity_summary(
            user_id=test_user_id,
            start_time=start_time,
            end_time=end_time,
        )

        # 3. Verify summary
        assert "total_count" in summary, "Summary should have total_count"
        assert summary["total_count"] >= 3, f"Expected at least 3 total, got {summary['total_count']}"

        assert "type_distribution" in summary, "Summary should have type_distribution"
        type_dist = summary["type_distribution"]

        # DataTypeEnum.CONVERSATION.value = "Conversation"
        assert "Conversation" in type_dist, "Should have Conversation type"
        assert type_dist["Conversation"] >= 3, f"Should have at least 3 CONVERSATION, got {type_dist['Conversation']}"

        assert "user_id" in summary, "Summary should have user_id"
        assert "time_range" in summary, "Summary should have time_range"

        logger.info(f"✅ Activity summary verified: {summary['total_count']} total")
        logger.info(f"   Type distribution: {type_dist}")

        # Cleanup
        for created in created_list:
            await repository.delete_by_event_id(str(created.id))


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_17_get_nonexistent_event_id(self, repository):
        """
        Test: get_by_event_id with non-existent ID
        Expected: Should return None
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_event_id (non-existent)")

        fake_id = "000000000000000000000000"
        result = await repository.get_by_event_id(fake_id)

        assert result is None, "Non-existent ID should return None"
        logger.info(f"✅ Non-existent ID handled correctly: returned None")

    async def test_18_delete_nonexistent_event_id(self, repository):
        """
        Test: delete_by_event_id with non-existent ID
        Expected: Should complete without error (return value depends on KV-Storage implementation)
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_event_id (non-existent)")

        fake_id = "000000000000000000000000"
        result = await repository.delete_by_event_id(fake_id)

        # Note: Return value depends on KV-Storage implementation
        # MongoDB won't find it, but KV-Storage might return True for delete operation
        assert isinstance(result, bool), "Should return a boolean"
        logger.info(f"✅ Non-existent ID deletion handled correctly: returned {result}")

    async def test_19_update_nonexistent_event_id(self, repository):
        """
        Test: update_by_event_id with non-existent ID
        Expected: Should return None
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: update_by_event_id (non-existent)")

        fake_id = "000000000000000000000000"
        update_data = {"summary": "Updated"}
        result = await repository.update_by_event_id(fake_id, update_data)

        assert result is None, "Updating non-existent ID should return None"
        logger.info(f"✅ Non-existent ID update handled correctly: returned None")

    async def test_20_verify_audit_fields(self, repository, test_user_id):
        """
        Test: Verify created_at and updated_at are set correctly
        Bug fix: After removing AuditBase from MemCellLite, these fields were not set
        """
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Verify created_at and updated_at fields")

        # 1. Create and append MemCell
        original = create_test_memcell(user_id=test_user_id, summary="Test audit fields")
        created = await repository.append_memcell(original)
        assert created is not None, "append_memcell should return MemCell"

        # 2. Verify audit fields are set after append
        assert created.created_at is not None, "❌ BUG: created_at should not be None!"
        assert created.updated_at is not None, "❌ BUG: updated_at should not be None!"
        logger.info(f"✅ After append: created_at={created.created_at}, updated_at={created.updated_at}")

        # 3. Retrieve from KV-Storage and verify persistence
        retrieved = await repository.get_by_event_id(str(created.id))
        assert retrieved is not None, "get_by_event_id should return MemCell"
        assert retrieved.created_at is not None, "❌ BUG: created_at should persist in KV-Storage!"
        assert retrieved.updated_at is not None, "❌ BUG: updated_at should persist in KV-Storage!"
        logger.info(f"✅ After retrieve: created_at={retrieved.created_at}, updated_at={retrieved.updated_at}")

        # 4. Verify timezones are consistent
        assert retrieved.created_at.tzinfo == retrieved.timestamp.tzinfo, "created_at timezone should match timestamp"
        assert retrieved.updated_at.tzinfo == retrieved.timestamp.tzinfo, "updated_at timezone should match timestamp"
        logger.info(f"✅ Timezone consistency verified: {retrieved.timestamp.tzinfo}")

        # 5. Verify created_at equals updated_at for newly created records
        time_diff = abs((retrieved.created_at - retrieved.updated_at).total_seconds())
        assert time_diff < 1, "created_at and updated_at should be nearly identical for new records"
        logger.info(f"✅ created_at ≈ updated_at (diff: {time_diff:.6f}s)")

        # Cleanup
        await repository.delete_by_event_id(str(created.id))
        logger.info("✅ Audit fields verification passed")


# ==================== Main Test Runner ====================


if __name__ == "__main__":
    """
    Run all tests with pytest

    Usage:
        pytest tests/test_memcell_crud_complete.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
