#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test MemCellRawRepository with DualStorageMixin (Model Proxy)

Verify that MongoDB Model 层拦截方案 works correctly for MemCell.
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
    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
        MemCellRawRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
        MemCellRawRepository,
    )
    return get_bean_by_type(MemCellRawRepository)


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


def create_test_memcell(user_id: str, summary: str = "Test memcell"):
    """Helper to create test MemCell"""
    from common_utils.datetime_utils import get_now_with_timezone
    from infra_layer.adapters.out.persistence.document.memory.memcell import (
        MemCell,
        DataTypeEnum,
    )

    return MemCell(
        user_id=user_id,
        timestamp=get_now_with_timezone(),
        summary=summary,
        group_id=f"group_{user_id}",
        participants=[user_id, "Participant1", "Participant2"],
        type=DataTypeEnum.CONVERSATION,
        subject=f"Subject: {summary}",
        keywords=["test", "memcell"],
        linked_entities=[f"entity_{uuid.uuid4().hex[:8]}"],
        extend={"test_flag": True},
    )


def get_logger():
    """Helper to get logger"""
    from core.observation.logger import get_logger as _get_logger
    return _get_logger(__name__)


class TestMemCellDualStorage:
    """Test MemCell Model Proxy 拦截方案"""

    async def test_01_insert_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: document.insert() is intercepted and syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell insert syncs to KV-Storage")

        # Create test data
        test_data = create_test_memcell(
            user_id=test_user_id,
            summary="Test memcell insert interception",
        )

        # Call Repository's append method (internally calls document.insert())
        created = await repository.append_memcell(test_data)
        assert created is not None, "append_memcell failed"
        assert created.id is not None, "ID should be set"
        doc_id = str(created.id)
        logger.info(f"✅ MemCell inserted via repository: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"
        logger.info(f"✅ Verified KV-Storage sync: {doc_id}")

        # Cleanup
        await repository.hard_delete_by_event_id(doc_id)
        logger.info("✅ Test passed")

    async def test_02_model_get_reads_from_kv(self, repository, kv_storage, test_user_id):
        """Test: model.get() is intercepted and reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell model.get() reads from KV-Storage")

        # Create test data
        test_data = create_test_memcell(user_id=test_user_id, summary="Test get from KV")
        created = await repository.append_memcell(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Call Repository method that uses model.get() internally
        retrieved = await repository.get_by_event_id(doc_id)
        assert retrieved is not None, "get_by_event_id failed"
        assert str(retrieved.id) == doc_id, "IDs don't match"
        assert retrieved.summary == created.summary, "Summaries don't match"
        logger.info(f"✅ Retrieved via model.get (KV interception): {doc_id}")

        # Cleanup
        await repository.hard_delete_by_event_id(doc_id)
        logger.info("✅ Test passed")

    async def test_03_model_find_works(self, repository, test_user_id):
        """Test: model.find() is intercepted correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell model.find() interception")

        # Create 3 test records
        created_ids = []
        for i in range(3):
            test_data = create_test_memcell(
                user_id=test_user_id,
                summary=f"Find test {i+1}",
            )
            created = await repository.append_memcell(test_data)
            created_ids.append(str(created.id))
            logger.info(f"✅ Created {i+1}/3: {created.id}")

        # Query using Repository method (internally uses model.find())
        results = await repository.find_by_user_id(user_id=test_user_id, limit=10)
        assert len(results) >= 3, f"Should find at least 3 records, got {len(results)}"
        logger.info(f"✅ Found {len(results)} records via model.find()")

        # Verify all created IDs are in results
        result_ids = {str(r.id) for r in results}
        for created_id in created_ids:
            assert created_id in result_ids, f"Created ID {created_id} not in results"
        logger.info("✅ All created records found in query results")

        # Cleanup
        for created_id in created_ids:
            await repository.hard_delete_by_event_id(created_id)
        logger.info("✅ Test passed")

    async def test_04_soft_delete_removes_from_kv(self, repository, kv_storage, test_user_id):
        """Test: soft delete (document.delete()) is intercepted and removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell soft delete removes from KV-Storage")

        # Create test data
        test_data = create_test_memcell(user_id=test_user_id, summary="Test soft delete")
        created = await repository.append_memcell(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Verify KV has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV should have data before delete"
        logger.info(f"✅ KV has data: {doc_id}")

        # Soft delete using Repository method (internally calls document.delete())
        deleted = await repository.delete_by_event_id(doc_id)
        assert deleted is True, "delete_by_event_id failed"
        logger.info(f"✅ Soft deleted from MongoDB: {doc_id}")

        # Verify KV-Storage is also deleted
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have data after soft delete"
        logger.info(f"✅ Verified KV-Storage deletion: {doc_id}")

        # Hard delete cleanup
        await repository.hard_delete_by_event_id(doc_id)

    async def test_05_update_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: document.save() (update) is intercepted and syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell update syncs to KV-Storage")

        # Create test data
        test_data = create_test_memcell(user_id=test_user_id, summary="Test update")
        created = await repository.append_memcell(test_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Update using Repository method (internally calls document.save())
        new_summary = "Updated summary"
        updated = await repository.update_by_event_id(
            event_id=doc_id,
            update_data={"summary": new_summary}
        )
        assert updated is not None, "update_by_event_id failed"
        assert updated.summary == new_summary, "Summary not updated"
        logger.info(f"✅ Updated MongoDB: {doc_id}")

        # Verify KV-Storage is updated
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"
        from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
        kv_doc = MemCell.model_validate_json(kv_value)
        assert kv_doc.summary == new_summary, "KV-Storage not updated"
        logger.info(f"✅ Verified KV-Storage update: {doc_id}")

        # Cleanup
        await repository.hard_delete_by_event_id(doc_id)
        logger.info("✅ Test passed")

    async def test_06_batch_operations(self, repository, test_user_id):
        """Test: Repository 的批量操作方法完全不需要改动"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell batch operations work unchanged")

        # Create 3 records
        created_ids = []
        for i in range(3):
            test_data = create_test_memcell(
                user_id=test_user_id,
                summary=f"Batch test {i+1}",
            )
            created = await repository.append_memcell(test_data)
            created_ids.append(str(created.id))
        logger.info(f"✅ Created 3 records")

        # Batch get
        results_dict = await repository.get_by_event_ids(created_ids)
        assert len(results_dict) == 3, f"Should get 3 records, got {len(results_dict)}"
        logger.info(f"✅ Batch get works: {len(results_dict)} records")

        # Cleanup
        for created_id in created_ids:
            await repository.hard_delete_by_event_id(created_id)
        logger.info("✅ Test passed - Repository batch methods unchanged")

    async def test_07_delete_by_user_id(self, repository, kv_storage):
        """Test: delete_by_user_id removes from both MongoDB and KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell delete_by_user_id")

        # Use unique user ID for this test
        test_user = f"test_user_delete_{uuid.uuid4().hex[:8]}"

        # Create 3 records
        created_ids = []
        for i in range(3):
            test_data = create_test_memcell(
                user_id=test_user,
                summary=f"Delete test {i+1}",
            )
            created = await repository.append_memcell(test_data)
            created_ids.append(str(created.id))
        logger.info(f"✅ Created 3 records for user: {test_user}")

        # Verify all in KV-Storage
        for doc_id in created_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None, f"KV should have {doc_id}"
        logger.info("✅ All records in KV-Storage")

        # Soft delete all by user_id
        deleted_count = await repository.delete_by_user_id(test_user)
        assert deleted_count >= 3, f"Expected to delete at least 3, deleted {deleted_count}"
        logger.info(f"✅ Soft deleted {deleted_count} records for user")

        # Verify MongoDB soft deleted (not visible in normal queries)
        results_after = await repository.find_by_user_id(user_id=test_user, limit=10)
        assert len(results_after) == 0, f"Expected 0 records after soft delete, got {len(results_after)}"
        logger.info(f"✅ Verified MongoDB soft deletion: count = 0")

        # Hard delete cleanup
        await repository.hard_delete_by_user_id(test_user)
        logger.info("✅ Test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
