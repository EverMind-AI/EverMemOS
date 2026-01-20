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

    async def test_08_hard_delete_removes_from_kv(self, repository, kv_storage, test_user_id):
        """Test: hard_delete_by_event_id physically deletes from both MongoDB and KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell hard_delete_by_event_id")

        # Create test data
        test_data = create_test_memcell(user_id=test_user_id, summary="Test hard delete")
        created = await repository.append_memcell(test_data)
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Verify KV has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV should have data before hard delete"
        logger.info(f"✅ KV has data: {doc_id}")

        # Hard delete (physical deletion)
        deleted = await repository.hard_delete_by_event_id(doc_id)
        assert deleted is True, "hard_delete_by_event_id failed"
        logger.info(f"✅ Hard deleted from MongoDB: {doc_id}")

        # Verify KV-Storage is also deleted
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have data after hard delete"
        logger.info(f"✅ Verified KV-Storage deletion after hard delete")

    async def test_09_restore_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: restore_by_event_id restores document and syncs back to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell restore_by_event_id")

        # Create test data
        test_data = create_test_memcell(user_id=test_user_id, summary="Test restore")
        created = await repository.append_memcell(test_data)
        doc_id = str(created.id)
        logger.info(f"✅ Created: {doc_id}")

        # Soft delete
        await repository.delete_by_event_id(doc_id)
        logger.info(f"✅ Soft deleted: {doc_id}")

        # Verify KV is deleted
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV should be deleted after soft delete"

        # Restore the document
        restored = await repository.restore_by_event_id(doc_id)
        assert restored is True, "restore_by_event_id failed"
        logger.info(f"✅ Restored document: {doc_id}")

        # Verify KV-Storage is restored
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV should be restored after restore"

        from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
        kv_doc = MemCell.model_validate_json(kv_value)
        assert str(kv_doc.id) == doc_id, "Restored KV document ID should match"
        logger.info(f"✅ Verified KV-Storage restoration")

        # Cleanup
        await repository.hard_delete_by_event_id(doc_id)
        logger.info("✅ Test passed")

    async def test_10_restore_by_user_id(self, repository, kv_storage):
        """Test: restore_by_user_id batch restores and syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell restore_by_user_id batch restore")

        # Use unique user ID
        test_user = f"test_user_restore_{uuid.uuid4().hex[:8]}"

        # Create 3 records
        created_ids = []
        for i in range(3):
            test_data = create_test_memcell(
                user_id=test_user,
                summary=f"Restore test {i+1}",
            )
            created = await repository.append_memcell(test_data)
            created_ids.append(str(created.id))
        logger.info(f"✅ Created 3 records for user: {test_user}")

        # Soft delete all by user_id
        deleted_count = await repository.delete_by_user_id(test_user)
        assert deleted_count >= 3, f"Expected to soft delete at least 3"
        logger.info(f"✅ Soft deleted {deleted_count} records")

        # Verify all deleted from KV
        for doc_id in created_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is None, f"KV should not have {doc_id} after soft delete"
        logger.info("✅ All records deleted from KV-Storage")

        # Restore all by user_id
        restored_count = await repository.restore_by_user_id(test_user)
        assert restored_count >= 3, f"Expected to restore at least 3, restored {restored_count}"
        logger.info(f"✅ Restored {restored_count} records for user")

        # Verify all restored to KV
        for doc_id in created_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None, f"KV should have {doc_id} after restore"
        logger.info("✅ All records restored to KV-Storage")

        # Hard delete cleanup
        await repository.hard_delete_by_user_id(test_user)
        logger.info("✅ Test passed")

    async def test_11_find_by_user_and_time_range(self, repository, test_user_id):
        """Test: find_by_user_and_time_range filters correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell find_by_user_and_time_range")

        from common_utils.datetime_utils import get_now_with_timezone
        from datetime import timedelta

        # Create records with different timestamps
        now = get_now_with_timezone()
        past_time = now - timedelta(hours=3)
        recent_time = now - timedelta(hours=1)

        # Create past record
        from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell, DataTypeEnum
        past_record = MemCell(
            user_id=test_user_id,
            timestamp=past_time,
            summary="Past memcell",
            type=DataTypeEnum.CONVERSATION,
        )
        created_past = await repository.append_memcell(past_record)
        logger.info(f"✅ Created past record at {past_time}")

        # Create recent record
        recent_record = MemCell(
            user_id=test_user_id,
            timestamp=recent_time,
            summary="Recent memcell",
            type=DataTypeEnum.CONVERSATION,
        )
        created_recent = await repository.append_memcell(recent_record)
        logger.info(f"✅ Created recent record at {recent_time}")

        # Query time range: last 2 hours
        start_time = now - timedelta(hours=2)
        end_time = now + timedelta(hours=1)
        results = await repository.find_by_user_and_time_range(
            user_id=test_user_id,
            start_time=start_time,
            end_time=end_time,
        )

        # Should only include recent record, not past record
        result_ids = {str(r.id) for r in results}
        assert str(created_recent.id) in result_ids, "Recent record should be in results"
        assert str(created_past.id) not in result_ids, "Past record should NOT be in results"
        logger.info(f"✅ Time range filter works correctly")

        # Cleanup
        await repository.hard_delete_by_event_id(str(created_past.id))
        await repository.hard_delete_by_event_id(str(created_recent.id))
        logger.info("✅ Test passed")

    async def test_12_find_by_group_id(self, repository, test_user_id):
        """Test: find_by_group_id filters by group"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell find_by_group_id")

        # Create records with different group_ids
        group1_id = f"group1_{uuid.uuid4().hex[:8]}"
        group2_id = f"group2_{uuid.uuid4().hex[:8]}"

        # Create 2 records for group1
        group1_ids = []
        for i in range(2):
            test_data = create_test_memcell(
                user_id=test_user_id,
                summary=f"Group1 memcell {i+1}"
            )
            test_data.group_id = group1_id
            created = await repository.append_memcell(test_data)
            group1_ids.append(str(created.id))

        # Create 1 record for group2
        test_data2 = create_test_memcell(
            user_id=test_user_id,
            summary="Group2 memcell"
        )
        test_data2.group_id = group2_id
        created2 = await repository.append_memcell(test_data2)
        logger.info(f"✅ Created records for 2 different groups")

        # Query by group1
        results = await repository.find_by_group_id(group_id=group1_id)
        result_ids = {str(r.id) for r in results}

        # Should only include group1 records
        assert len(results) >= 2, f"Should find at least 2 group1 records"
        for group1_id_str in group1_ids:
            assert group1_id_str in result_ids, f"Group1 record {group1_id_str} should be in results"
        assert str(created2.id) not in result_ids, "Group2 record should NOT be in results"
        logger.info(f"✅ Group filter works correctly")

        # Cleanup
        for doc_id in group1_ids:
            await repository.hard_delete_by_event_id(doc_id)
        await repository.hard_delete_by_event_id(str(created2.id))
        logger.info("✅ Test passed")

    async def test_13_find_by_participants(self, repository, test_user_id):
        """Test: find_by_participants filters by participant list"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell find_by_participants")

        # Create records with different participants
        participant1 = f"Alice_{uuid.uuid4().hex[:4]}"
        participant2 = f"Bob_{uuid.uuid4().hex[:4]}"
        participant3 = f"Charlie_{uuid.uuid4().hex[:4]}"

        # Create record with Alice and Bob
        test_data1 = create_test_memcell(
            user_id=test_user_id,
            summary="Alice and Bob conversation"
        )
        test_data1.participants = [participant1, participant2]
        created1 = await repository.append_memcell(test_data1)

        # Create record with Alice and Charlie
        test_data2 = create_test_memcell(
            user_id=test_user_id,
            summary="Alice and Charlie conversation"
        )
        test_data2.participants = [participant1, participant3]
        created2 = await repository.append_memcell(test_data2)
        logger.info(f"✅ Created records with different participants")

        # Query by participants including Alice and Bob (match_all=True)
        results = await repository.find_by_participants(
            participants=[participant1, participant2],
            match_all=True
        )
        result_ids = {str(r.id) for r in results}

        # Should only include record with both Alice AND Bob
        assert str(created1.id) in result_ids, "Alice+Bob record should be in results"
        assert str(created2.id) not in result_ids, "Alice+Charlie record should NOT be in results (match_all=True)"
        logger.info(f"✅ Participant filter (match_all=True) works correctly")

        # Query by participants including Alice OR Bob (match_all=False)
        results2 = await repository.find_by_participants(
            participants=[participant1, participant2],
            match_all=False
        )
        result_ids2 = {str(r.id) for r in results2}

        # Should include both records (Alice is in both)
        assert str(created1.id) in result_ids2, "Alice+Bob record should be in results"
        assert str(created2.id) in result_ids2, "Alice+Charlie record should also be in results (match_all=False)"
        logger.info(f"✅ Participant filter (match_all=False) works correctly")

        # Cleanup
        await repository.hard_delete_by_event_id(str(created1.id))
        await repository.hard_delete_by_event_id(str(created2.id))
        logger.info("✅ Test passed")

    async def test_14_search_by_keywords(self, repository, test_user_id):
        """Test: search_by_keywords filters by keyword list"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell search_by_keywords")

        # Create records with different keywords
        test_data1 = create_test_memcell(
            user_id=test_user_id,
            summary="Python programming discussion"
        )
        test_data1.keywords = ["python", "programming"]
        created1 = await repository.append_memcell(test_data1)

        test_data2 = create_test_memcell(
            user_id=test_user_id,
            summary="Python data science"
        )
        test_data2.keywords = ["python", "data-science"]
        created2 = await repository.append_memcell(test_data2)
        logger.info(f"✅ Created records with different keywords")

        # Search by keywords: python AND programming (match_all=True)
        results = await repository.search_by_keywords(
            keywords=["python", "programming"],
            match_all=True
        )
        result_ids = {str(r.id) for r in results}

        # Should only include first record
        assert str(created1.id) in result_ids, "Python+programming record should be in results"
        assert str(created2.id) not in result_ids, "Python+data-science should NOT be in results (match_all=True)"
        logger.info(f"✅ Keyword search (match_all=True) works correctly")

        # Cleanup
        await repository.hard_delete_by_event_id(str(created1.id))
        await repository.hard_delete_by_event_id(str(created2.id))
        logger.info("✅ Test passed")

    async def test_15_count_by_user_id(self, repository, test_user_id):
        """Test: count_by_user_id returns correct count"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell count_by_user_id")

        # Use unique user ID
        test_user = f"test_user_count_{uuid.uuid4().hex[:8]}"

        # Initially should be 0
        initial_count = await repository.count_by_user_id(test_user)
        assert initial_count == 0, f"Initial count should be 0, got {initial_count}"
        logger.info(f"✅ Initial count: {initial_count}")

        # Create 5 records
        created_ids = []
        for i in range(5):
            test_data = create_test_memcell(
                user_id=test_user,
                summary=f"Count test {i+1}"
            )
            created = await repository.append_memcell(test_data)
            created_ids.append(str(created.id))

        # Count should be 5
        after_count = await repository.count_by_user_id(test_user)
        assert after_count == 5, f"Count should be 5, got {after_count}"
        logger.info(f"✅ After creation count: {after_count}")

        # Soft delete 2 records
        await repository.delete_by_event_id(created_ids[0])
        await repository.delete_by_event_id(created_ids[1])

        # Count should be 3 (soft deleted are excluded)
        after_delete_count = await repository.count_by_user_id(test_user)
        assert after_delete_count == 3, f"Count should be 3 after soft delete, got {after_delete_count}"
        logger.info(f"✅ After soft delete count: {after_delete_count}")

        # Cleanup
        await repository.hard_delete_by_user_id(test_user)
        logger.info("✅ Test passed")

    async def test_16_get_latest_by_user(self, repository, test_user_id):
        """Test: get_latest_by_user returns most recent records"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemCell get_latest_by_user")

        from common_utils.datetime_utils import get_now_with_timezone
        from datetime import timedelta

        now = get_now_with_timezone()

        # Create 5 records with different timestamps
        created_ids = []
        for i in range(5):
            from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell, DataTypeEnum
            timestamp = now - timedelta(hours=i)  # Each hour older
            test_data = MemCell(
                user_id=test_user_id,
                timestamp=timestamp,
                summary=f"Record {i+1} from {i} hours ago",
                type=DataTypeEnum.CONVERSATION,
            )
            created = await repository.append_memcell(test_data)
            created_ids.append(str(created.id))
        logger.info(f"✅ Created 5 records with different timestamps")

        # Get latest 3
        latest = await repository.get_latest_by_user(user_id=test_user_id, limit=3)
        assert len(latest) == 3, f"Should get 3 latest records, got {len(latest)}"

        # Verify order (most recent first)
        # The first record (index 0) should be the most recent
        assert str(latest[0].id) == created_ids[0], "First result should be the most recent"
        logger.info(f"✅ get_latest_by_user returns correct order")

        # Cleanup
        for doc_id in created_ids:
            await repository.hard_delete_by_event_id(doc_id)
        logger.info("✅ Test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
