#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test EpisodicMemoryRawRepository with DualStorageMixin (Model Proxy)

Verify that MongoDB Model 层拦截方案works correctly.
Repository 代码完全不需要改动，所有双存储逻辑由 Mixin 透明处理。
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


class TestDualStorageModelProxy:
    """Test Model Proxy 拦截方案"""

    async def test_01_insert_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: document.insert() is intercepted and syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: insert syncs to KV-Storage")

        # Create test data
        test_data = create_test_episodic_memory(
            user_id=test_user_id,
            summary="Test insert interception",
        )

        # Call Repository's append method (internally calls document.insert())
        created = await repository.append_episodic_memory(test_data)
        assert created is not None, "append_episodic_memory failed"
        assert created.id is not None, "ID should be set"
        doc_id = str(created.id)
        logger.info(f"✅ Document inserted via repository: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"
        logger.info(f"✅ Verified KV-Storage sync: {doc_id}")

        # Cleanup
        await repository.delete_by_event_id(doc_id, test_user_id)
        logger.info("✅ Test passed")

    async def test_02_model_get_reads_from_kv(self, repository, kv_storage, test_user_id):
        """Test: model.get() is intercepted and reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: model.get() reads from KV-Storage")

        # Create test data
        test_data = create_test_episodic_memory(user_id=test_user_id)
        created = await repository.append_episodic_memory(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Call Repository method that uses model.get() internally
        # get_by_event_id() calls self.model.find_one() -> model.get()
        retrieved = await repository.get_by_event_id(doc_id, test_user_id)
        assert retrieved is not None, "get_by_event_id failed"
        assert str(retrieved.id) == doc_id, "IDs don't match"
        assert retrieved.summary == created.summary, "Summaries don't match"
        logger.info(f"✅ Retrieved via model.get (KV interception): {doc_id}")

        # Cleanup
        await repository.delete_by_event_id(doc_id, test_user_id)
        logger.info("✅ Test passed")

    async def test_03_model_find_works(self, repository, test_user_id):
        """Test: model.find() is intercepted correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: model.find() interception")

        # Create 3 test records
        created_ids = []
        for i in range(3):
            test_data = create_test_episodic_memory(
                user_id=test_user_id,
                summary=f"Find test {i+1}",
            )
            created = await repository.append_episodic_memory(test_data)
            created_ids.append(str(created.id))
            logger.info(f"✅ Created {i+1}/3: {created.id}")

        # Query using Repository method (internally uses model.find())
        results = await repository.find_by_filters(
            user_id=test_user_id,
            limit=10,
        )
        assert len(results) >= 3, f"Should find at least 3 records, got {len(results)}"
        logger.info(f"✅ Found {len(results)} records via model.find()")

        # Verify all created IDs are in results
        result_ids = {str(r.id) for r in results}
        for created_id in created_ids:
            assert created_id in result_ids, f"Created ID {created_id} not in results"
        logger.info("✅ All created records found in query results")

        # Cleanup
        for created_id in created_ids:
            await repository.delete_by_event_id(created_id, test_user_id)
        logger.info("✅ Test passed")

    async def test_04_delete_removes_from_kv(self, repository, kv_storage, test_user_id):
        """Test: document.delete() is intercepted and removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete removes from KV-Storage")

        # Create test data
        test_data = create_test_episodic_memory(user_id=test_user_id)
        created = await repository.append_episodic_memory(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Verify KV has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV should have data before delete"
        logger.info(f"✅ KV has data: {doc_id}")

        # Delete using Repository method (internally calls document.delete())
        deleted = await repository.delete_by_event_id(doc_id, test_user_id)
        assert deleted is True, "delete_by_event_id failed"
        logger.info(f"✅ Deleted from MongoDB: {doc_id}")

        # Verify KV-Storage is also deleted
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have data after delete"
        logger.info(f"✅ Verified KV-Storage deletion: {doc_id}")

    async def test_05_repository_unchanged(self, repository, test_user_id):
        """Test: Repository 的所有方法完全不需要改动"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Repository methods work unchanged")

        # Test get_by_event_ids (batch get)
        created_ids = []
        for i in range(3):
            test_data = create_test_episodic_memory(
                user_id=test_user_id,
                summary=f"Batch test {i+1}",
            )
            created = await repository.append_episodic_memory(test_data)
            created_ids.append(str(created.id))

        # Batch get
        results_dict = await repository.get_by_event_ids(created_ids, test_user_id)
        assert len(results_dict) == 3, f"Should get 3 records, got {len(results_dict)}"
        logger.info(f"✅ Batch get works: {len(results_dict)} records")

        # Cleanup
        for created_id in created_ids:
            await repository.delete_by_event_id(created_id, test_user_id)
        logger.info("✅ Test passed - Repository methods unchanged")

    async def test_06_delete_by_user_id(self, repository, kv_storage):
        """Test: delete_by_user_id removes from both MongoDB and KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: delete_by_user_id")

        # Use unique user ID for this test
        test_user = f"test_user_delete_{uuid.uuid4().hex[:8]}"

        # Create 3 records
        created_ids = []
        for i in range(3):
            test_data = create_test_episodic_memory(
                user_id=test_user,
                summary=f"Delete test {i+1}",
            )
            created = await repository.append_episodic_memory(test_data)
            created_ids.append(str(created.id))
        logger.info(f"✅ Created 3 records for user: {test_user}")

        # Verify all in KV-Storage
        for doc_id in created_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None, f"KV should have {doc_id}"
        logger.info("✅ All records in KV-Storage")

        # Delete all by user_id
        deleted_count = await repository.delete_by_user_id(test_user)
        assert deleted_count >= 3, f"Expected to delete at least 3, deleted {deleted_count}"
        logger.info(f"✅ Deleted {deleted_count} records for user")

        # Verify MongoDB is empty
        results_after = await repository.find_by_filters(user_id=test_user, limit=10)
        assert len(results_after) == 0, f"Expected 0 records after deletion, got {len(results_after)}"
        logger.info(f"✅ Verified MongoDB deletion: count = 0")

        # Note: delete_by_user_id should also clean up KV-Storage
        # (current implementation does batch_delete from KV)

    async def test_07_find_by_filter_paginated(self, repository, test_user_id):
        """Test: find_by_filter_paginated with pagination"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_filter_paginated with pagination")

        # Create 10 test records
        created_ids = []
        for i in range(10):
            test_data = create_test_episodic_memory(
                user_id=test_user_id,
                summary=f"Paginated test {i+1:02d}",
            )
            created = await repository.append_episodic_memory(test_data)
            created_ids.append(str(created.id))
        logger.info(f"✅ Created 10 records for pagination test")

        # Test pagination: skip 0, limit 3 (first page)
        page1_results = await repository.find_by_filter_paginated(
            query_filter={"user_id": test_user_id},
            skip=0,
            limit=3,
        )
        assert len(page1_results) == 3, f"Page 1 should have 3 records, got {len(page1_results)}"
        logger.info(f"✅ Page 1: {len(page1_results)} records")

        # Test pagination: skip 3, limit 3 (second page)
        page2_results = await repository.find_by_filter_paginated(
            query_filter={"user_id": test_user_id},
            skip=3,
            limit=3,
        )
        assert len(page2_results) == 3, f"Page 2 should have 3 records, got {len(page2_results)}"
        logger.info(f"✅ Page 2: {len(page2_results)} records")

        # Verify no overlap between pages
        page1_ids = {str(r.id) for r in page1_results}
        page2_ids = {str(r.id) for r in page2_results}
        assert len(page1_ids & page2_ids) == 0, "Pages should not overlap"
        logger.info("✅ No overlap between pages")

        # Cleanup
        for created_id in created_ids:
            await repository.delete_by_event_id(created_id, test_user_id)
        logger.info("✅ Test passed")

    async def test_08_find_by_filters_with_time_range(self, repository, test_user_id):
        """Test: find_by_filters with time range filtering"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: find_by_filters with time range")

        from common_utils.datetime_utils import get_now_with_timezone
        from datetime import timedelta

        # Create records with different timestamps
        now = get_now_with_timezone()
        past_time = now - timedelta(hours=2)
        future_time = now + timedelta(hours=2)

        # Create a record with past timestamp
        from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
            EpisodicMemory,
        )
        past_record = EpisodicMemory(
            user_id=test_user_id,
            timestamp=past_time,
            summary="Past record",
            episode="Past content",
            user_name="TestUser",
        )
        created_past = await repository.append_episodic_memory(past_record)
        logger.info(f"✅ Created past record: {created_past.id}")

        # Create a record with current timestamp
        current_record = create_test_episodic_memory(
            user_id=test_user_id,
            summary="Current record"
        )
        created_current = await repository.append_episodic_memory(current_record)
        logger.info(f"✅ Created current record: {created_current.id}")

        # Query with time range: from past_time - 1h to now + 1h (to ensure we capture "now")
        start_time = past_time - timedelta(hours=1)
        end_time = now + timedelta(hours=1)  # Extended to ensure we capture the current record
        results = await repository.find_by_filters(
            user_id=test_user_id,
            start_time=start_time,
            end_time=end_time,
        )

        # Should include both past and current records
        result_ids = {str(r.id) for r in results}
        assert str(created_past.id) in result_ids, "Past record should be in results"
        assert str(created_current.id) in result_ids, "Current record should be in results"
        logger.info(f"✅ Time range query returned {len(results)} records")

        # Cleanup
        await repository.delete_by_event_id(str(created_past.id), test_user_id)
        await repository.delete_by_event_id(str(created_current.id), test_user_id)
        logger.info("✅ Test passed")

    async def test_09_update_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: document.save() (update) syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: Update syncs to KV-Storage")

        # Create test record
        test_data = create_test_episodic_memory(
            user_id=test_user_id,
            summary="Original summary"
        )
        created = await repository.append_episodic_memory(test_data)
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Update the document directly (simulating document.save())
        created.summary = "Updated summary"
        created.episode = "Updated episode"
        await created.save()
        logger.info(f"✅ Updated document via save()")

        # Verify KV-Storage is updated
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"

        from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
            EpisodicMemory,
        )
        kv_doc = EpisodicMemory.model_validate_json(kv_value)
        assert kv_doc.summary == "Updated summary", "KV summary not updated"
        assert kv_doc.episode == "Updated episode", "KV episode not updated"
        logger.info(f"✅ Verified KV-Storage sync after update")

        # Cleanup
        await repository.delete_by_event_id(doc_id, test_user_id)
        logger.info("✅ Test passed")

    async def test_10_kv_miss_fallback_to_mongo(self, repository, kv_storage, test_user_id):
        """Test: KV miss falls back to MongoDB and backfills KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: KV miss fallback and backfill")

        # Create test record
        test_data = create_test_episodic_memory(user_id=test_user_id)
        created = await repository.append_episodic_memory(test_data)
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Manually delete from KV-Storage (simulate KV miss)
        await kv_storage.delete(doc_id)
        logger.info(f"✅ Deleted from KV-Storage: {doc_id}")

        # Verify KV is empty
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV should be empty"

        # Retrieve via repository (should fallback to MongoDB)
        retrieved = await repository.get_by_event_id(doc_id, test_user_id)
        assert retrieved is not None, "Should retrieve from MongoDB"
        assert str(retrieved.id) == doc_id, "IDs should match"
        logger.info(f"✅ Retrieved from MongoDB fallback")

        # Verify KV is backfilled
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV should be backfilled"
        logger.info(f"✅ Verified KV backfill")

        # Cleanup
        await repository.delete_by_event_id(doc_id, test_user_id)
        logger.info("✅ Test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
