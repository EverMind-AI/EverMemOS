#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test EpisodicMemoryRawRepository with DualStorageMixin

Verify that the minimal-invasive dual storage implementation works correctly.
"""

import asyncio
import pytest
import pytest_asyncio
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
        EpisodicMemoryRawRepository,
    )
    return get_bean_by_type(EpisodicMemoryRawRepository)


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )
    return get_bean_by_type(KVStorageInterface)


@pytest.fixture
def test_user_id():
    """Generate unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


def create_test_episodic_memory(user_id: str, summary: str = "Test memory"):
    """Helper to create test EpisodicMemory"""
    from common_utils.datetime_utils import get_now_with_timezone
    from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
        EpisodicMemory,
    )

    return EpisodicMemory(
        user_id=user_id,
        timestamp=get_now_with_timezone(),
        summary=summary,
        episode="Test episode content",
        user_name=f"TestUser_{user_id[-8:]}",
        group_id=f"group_{user_id}",
        group_name=f"TestGroup",
        participants=[user_id, "Participant1"],
        type="Conversation",
        subject=f"Subject: {summary}",
        keywords=["test", "memory"],
        linked_entities=[f"entity_{uuid.uuid4().hex[:8]}"],
        extend={"test_flag": True},
    )


def get_logger():
    """Helper to get logger"""
    from core.observation.logger import get_logger as _get_logger
    return _get_logger(__name__)


class TestDualStorageMixin:
    """Test DualStorageMixin functionality"""

    async def test_01_append_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: append automatically syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: append syncs to KV-Storage")

        # Create test data
        test_data = create_test_episodic_memory(
            user_id=test_user_id,
            summary="Test memory for append",
        )

        # Append
        created = await repository.append(test_data)
        assert created is not None, "append failed"
        assert created.id is not None, "ID should be set"
        doc_id = str(created.id)
        logger.info(f"✅ Appended to MongoDB: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"
        logger.info(f"✅ Verified KV-Storage sync: {doc_id}")

        # Cleanup
        await repository.delete_by_id(doc_id)
        logger.info("✅ Test passed")

    async def test_02_get_by_id_reads_from_kv(self, repository, kv_storage, test_user_id):
        """Test: get_by_id reads from KV-Storage (fast path)"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_id reads from KV-Storage")

        # Create test data
        test_data = create_test_episodic_memory(user_id=test_user_id)
        created = await repository.append(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Get by ID (should read from KV)
        retrieved = await repository.get_by_id(doc_id)
        assert retrieved is not None, "get_by_id failed"
        assert str(retrieved.id) == doc_id, "IDs don't match"
        assert retrieved.summary == created.summary, "Summaries don't match"
        logger.info(f"✅ Retrieved from KV: {doc_id}")

        # Cleanup
        await repository.delete_by_id(doc_id)
        logger.info("✅ Test passed")

    async def test_03_update_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: update_by_id syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: update_by_id syncs to KV-Storage")

        # Create test data
        test_data = create_test_episodic_memory(user_id=test_user_id)
        created = await repository.append(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Update
        update_data = {"summary": "Updated summary", "keywords": ["updated", "test"]}
        updated = await repository.update_by_id(doc_id, update_data)
        assert updated is not None, "update_by_id failed"
        assert updated.summary == "Updated summary", "Summary not updated"
        logger.info(f"✅ Updated MongoDB: {doc_id}")

        # Verify KV-Storage is updated
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"
        from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
            EpisodicMemory,
        )
        kv_doc = EpisodicMemory.model_validate_json(kv_value)
        assert kv_doc.summary == "Updated summary", "KV-Storage not updated"
        logger.info(f"✅ Verified KV-Storage update: {doc_id}")

        # Cleanup
        await repository.delete_by_id(doc_id)
        logger.info("✅ Test passed")

    async def test_04_delete_removes_from_kv(self, repository, kv_storage, test_user_id):
        """Test: delete_by_id removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_id removes from KV-Storage")

        # Create test data
        test_data = create_test_episodic_memory(user_id=test_user_id)
        created = await repository.append(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Verify KV has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV should have data before delete"

        # Delete
        deleted = await repository.delete_by_id(doc_id)
        assert deleted is True, "delete_by_id failed"
        logger.info(f"✅ Deleted from MongoDB: {doc_id}")

        # Verify KV-Storage is also deleted
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have data after delete"
        logger.info(f"✅ Verified KV-Storage deletion: {doc_id}")

    async def test_05_find_by_filter(self, repository, test_user_id):
        """Test: find_by_filter works correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_filter")

        # Create 3 test records
        created_ids = []
        for i in range(3):
            test_data = create_test_episodic_memory(
                user_id=test_user_id,
                summary=f"Filter test {i+1}",
            )
            created = await repository.append(test_data)
            created_ids.append(str(created.id))
            logger.info(f"✅ Created {i+1}/3: {created.id}")

        # Query by user_id
        results = await repository.find_by_filter(
            query_filter={"user_id": test_user_id},
            limit=10,
        )
        assert len(results) >= 3, f"Should find at least 3 records, got {len(results)}"
        logger.info(f"✅ Found {len(results)} records by filter")

        # Verify all created IDs are in results
        result_ids = {str(r.id) for r in results}
        for created_id in created_ids:
            assert created_id in result_ids, f"Created ID {created_id} not in results"
        logger.info("✅ All created records found in query results")

        # Cleanup
        for created_id in created_ids:
            await repository.delete_by_id(created_id)
        logger.info("✅ Test passed")

    async def test_06_get_by_event_id(self, repository, test_user_id):
        """Test: get_by_event_id (domain-specific method)"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: get_by_event_id")

        # Create test data
        test_data = create_test_episodic_memory(user_id=test_user_id)
        created = await repository.append(test_data)
        assert created is not None
        event_id = str(created.id)
        logger.info(f"✅ Created: {event_id}")

        # Get by event_id with correct user_id
        retrieved = await repository.get_by_event_id(event_id, test_user_id)
        assert retrieved is not None, "get_by_event_id failed"
        assert str(retrieved.id) == event_id, "IDs don't match"
        logger.info(f"✅ Retrieved by event_id: {event_id}")

        # Try with wrong user_id
        wrong_user = "wrong_user_123"
        retrieved_wrong = await repository.get_by_event_id(event_id, wrong_user)
        assert retrieved_wrong is None, "Should return None for wrong user_id"
        logger.info(f"✅ Correctly rejected wrong user_id")

        # Cleanup
        await repository.delete_by_id(event_id)
        logger.info("✅ Test passed")

    async def test_07_find_by_filters(self, repository, test_user_id):
        """Test: find_by_filters (domain-specific method)"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_filters")

        # Create test data
        created_ids = []
        for i in range(3):
            test_data = create_test_episodic_memory(
                user_id=test_user_id,
                summary=f"Filters test {i+1}",
            )
            created = await repository.append(test_data)
            created_ids.append(str(created.id))
            logger.info(f"✅ Created {i+1}/3: {created.id}")

        # Query using find_by_filters
        results = await repository.find_by_filters(
            user_id=test_user_id,
            limit=10,
        )
        assert len(results) >= 3, f"Should find at least 3 records, got {len(results)}"
        logger.info(f"✅ Found {len(results)} records using find_by_filters")

        # Cleanup
        for created_id in created_ids:
            await repository.delete_by_id(created_id)
        logger.info("✅ Test passed")

    async def test_08_delete_by_user_id(self, repository):
        """Test: delete_by_user_id (domain-specific method)"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_user_id")

        # Use unique user ID for this test
        test_user = f"test_user_delete_{uuid.uuid4().hex[:8]}"

        # Create 3 records
        for i in range(3):
            test_data = create_test_episodic_memory(
                user_id=test_user,
                summary=f"Delete test {i+1}",
            )
            await repository.append(test_data)
        logger.info(f"✅ Created 3 records for user: {test_user}")

        # Verify count before deletion
        results_before = await repository.find_by_filter(
            query_filter={"user_id": test_user}
        )
        count_before = len(results_before)
        assert count_before >= 3, f"Expected at least 3 records, got {count_before}"
        logger.info(f"✅ Count before deletion: {count_before}")

        # Delete all by user_id
        deleted_count = await repository.delete_by_user_id(test_user)
        assert deleted_count >= 3, f"Expected to delete at least 3, deleted {deleted_count}"
        logger.info(f"✅ Deleted {deleted_count} records for user")

        # Verify count after deletion
        results_after = await repository.find_by_filter(
            query_filter={"user_id": test_user}
        )
        count_after = len(results_after)
        assert count_after == 0, f"Expected 0 records after deletion, got {count_after}"
        logger.info(f"✅ Verified deletion: count = 0")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
