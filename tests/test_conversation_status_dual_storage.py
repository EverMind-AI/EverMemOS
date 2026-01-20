#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ConversationStatusRawRepository with DualStorageMixin

Verify that DualStorageMixin works correctly for ConversationStatus.
Repository code remains unchanged, all dual storage logic is handled transparently by Mixin.
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from core.observation.logger import get_logger

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
        ConversationStatusRawRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.conversation_status_raw_repository import (
        ConversationStatusRawRepository,
    )
    return get_bean_by_type(ConversationStatusRawRepository)


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


def create_test_conversation_status(
    group_id: str,
    old_msg_start_time: datetime = None,
    new_msg_start_time: datetime = None,
    last_memcell_time: datetime = None,
):
    """Create test ConversationStatus"""
    from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
        ConversationStatus,
    )

    if old_msg_start_time is None:
        old_msg_start_time = datetime.now() - timedelta(days=7)
    if new_msg_start_time is None:
        new_msg_start_time = datetime.now() - timedelta(days=1)
    if last_memcell_time is None:
        last_memcell_time = datetime.now()

    return ConversationStatus(
        group_id=group_id,
        old_msg_start_time=old_msg_start_time,
        new_msg_start_time=new_msg_start_time,
        last_memcell_time=last_memcell_time,
    )


@pytest.mark.asyncio
class TestConversationStatusDualStorage:
    """Test ConversationStatus dual storage functionality"""

    async def test_01_upsert_by_group_id_creates_and_syncs_to_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: upsert_by_group_id() creates new record and syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus upsert_by_group_id() creates and syncs to KV")

        # Create new record via upsert
        update_data = {
            "old_msg_start_time": datetime.now() - timedelta(days=7),
            "new_msg_start_time": datetime.now() - timedelta(days=1),
            "last_memcell_time": datetime.now(),
        }
        result = await repository.upsert_by_group_id(test_group_id, update_data)

        assert result is not None, "upsert_by_group_id failed"
        doc_id = str(result.id)
        logger.info(f"✅ Created: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"

        # Verify full data in KV
        from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
            ConversationStatus,
        )

        kv_doc = ConversationStatus.model_validate_json(kv_value)
        assert kv_doc.group_id == test_group_id
        assert kv_doc.old_msg_start_time is not None
        logger.info("✅ Test passed: upsert_by_group_id() creates and syncs to KV")

    async def test_02_upsert_by_group_id_updates_and_syncs_to_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: upsert_by_group_id() updates existing record and syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus upsert_by_group_id() updates and syncs to KV")

        # Create initial record
        update_data = {
            "old_msg_start_time": datetime.now() - timedelta(days=7),
            "new_msg_start_time": datetime.now() - timedelta(days=1),
        }
        result1 = await repository.upsert_by_group_id(test_group_id, update_data)
        assert result1 is not None
        doc_id = str(result1.id)

        # Update the record
        new_time = datetime.now(timezone.utc)
        update_data2 = {
            "last_memcell_time": new_time,
        }
        result2 = await repository.upsert_by_group_id(test_group_id, update_data2)

        assert result2 is not None, "upsert_by_group_id update failed"
        assert str(result2.id) == doc_id, "Should be the same document"
        # Compare timestamps (allow small differences due to serialization)
        assert abs((result2.last_memcell_time - new_time).total_seconds()) < 1

        # Verify KV-Storage has updated data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
            ConversationStatus,
        )

        kv_doc = ConversationStatus.model_validate_json(kv_value)
        assert kv_doc.last_memcell_time == new_time
        logger.info("✅ Test passed: upsert_by_group_id() updates and syncs to KV")

    async def test_03_get_by_group_id_reads_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: get_by_group_id() reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus get_by_group_id() reads from KV")

        # Create test data via upsert
        update_data = {
            "old_msg_start_time": datetime.now() - timedelta(days=7),
            "new_msg_start_time": datetime.now() - timedelta(days=1),
            "last_memcell_time": datetime.now(),
        }
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None

        # Get by group_id
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None, "get_by_group_id failed"
        assert retrieved.group_id == test_group_id
        assert retrieved.old_msg_start_time is not None
        logger.info("✅ Test passed: get_by_group_id() reads from KV")

    async def test_04_delete_by_group_id_removes_from_kv(
        self, repository, kv_storage, test_group_id
    ):
        """Test: delete_by_group_id() removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus delete_by_group_id() removes from KV")

        # Create test data via upsert
        update_data = {
            "old_msg_start_time": datetime.now() - timedelta(days=7),
        }
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        doc_id = str(created.id)

        # Verify KV has data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        # Delete by group_id
        success = await repository.delete_by_group_id(test_group_id)
        assert success, "delete_by_group_id should return True"

        # Verify KV removed
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have the data after delete"

        logger.info("✅ Test passed: delete_by_group_id() removes from KV")

    async def test_05_count_by_group_id_works(
        self, repository, kv_storage, test_group_id
    ):
        """Test: count_by_group_id() works correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus count_by_group_id() works")

        # Initially should be 0
        count = await repository.count_by_group_id(test_group_id)
        assert count == 0, f"Initial count should be 0, got {count}"

        # Create a record
        update_data = {
            "old_msg_start_time": datetime.now() - timedelta(days=7),
        }
        await repository.upsert_by_group_id(test_group_id, update_data)

        # Count should be 1 (group_id is unique, so max is 1)
        count = await repository.count_by_group_id(test_group_id)
        assert count == 1, f"Count should be 1 after creation, got {count}"

        logger.info("✅ Test passed: count_by_group_id() works")

    async def test_06_concurrent_upsert_handling(
        self, repository, kv_storage, test_group_id
    ):
        """Test: upsert_by_group_id() handles concurrent creation correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus concurrent upsert handling")

        # First upsert
        update_data1 = {
            "old_msg_start_time": datetime.now() - timedelta(days=7),
        }
        result1 = await repository.upsert_by_group_id(test_group_id, update_data1)
        assert result1 is not None

        # Second upsert (should update existing)
        update_data2 = {
            "new_msg_start_time": datetime.now() - timedelta(days=1),
        }
        result2 = await repository.upsert_by_group_id(test_group_id, update_data2)
        assert result2 is not None

        # Should be the same document
        assert str(result1.id) == str(result2.id), "Should update same document"

        # Verify final state in KV
        kv_value = await kv_storage.get(str(result2.id))
        assert kv_value is not None

        from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
            ConversationStatus,
        )

        kv_doc = ConversationStatus.model_validate_json(kv_value)
        assert kv_doc.old_msg_start_time is not None  # From first upsert
        assert kv_doc.new_msg_start_time is not None  # From second upsert

        logger.info("✅ Test passed: Concurrent upsert handled correctly")

    async def test_07_delete_nonexistent_group(self, repository, kv_storage):
        """Test: delete_by_group_id() handles nonexistent group gracefully"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus delete nonexistent group")

        # Try to delete non-existent group
        fake_group_id = f"nonexistent_{uuid.uuid4().hex[:8]}"
        success = await repository.delete_by_group_id(fake_group_id)
        assert not success, "delete_by_group_id should return False for nonexistent group"

        logger.info("✅ Test passed: Handles nonexistent group deletion gracefully")

    async def test_08_get_nonexistent_group(self, repository, kv_storage):
        """Test: get_by_group_id() handles nonexistent group gracefully"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus get nonexistent group")

        # Try to get non-existent group
        fake_group_id = f"nonexistent_{uuid.uuid4().hex[:8]}"
        result = await repository.get_by_group_id(fake_group_id)
        assert result is None, "get_by_group_id should return None for nonexistent group"

        logger.info("✅ Test passed: Handles nonexistent group get gracefully")

    async def test_09_full_lifecycle(self, repository, kv_storage, test_group_id):
        """Test: Full lifecycle - create, read, update, delete"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: ConversationStatus full lifecycle")

        # 1. Create
        update_data = {
            "old_msg_start_time": datetime.now() - timedelta(days=7),
        }
        created = await repository.upsert_by_group_id(test_group_id, update_data)
        assert created is not None
        doc_id = str(created.id)
        logger.info(f"Step 1: Created {doc_id}")

        # 2. Read
        retrieved = await repository.get_by_group_id(test_group_id)
        assert retrieved is not None
        assert str(retrieved.id) == doc_id
        logger.info("Step 2: Retrieved successfully")

        # 3. Update
        new_time = datetime.now(timezone.utc)
        update_data2 = {
            "new_msg_start_time": new_time,
        }
        updated = await repository.upsert_by_group_id(test_group_id, update_data2)
        assert updated is not None
        assert str(updated.id) == doc_id
        # Compare timestamps (allow small differences due to serialization)
        assert abs((updated.new_msg_start_time - new_time).total_seconds()) < 1
        logger.info("Step 3: Updated successfully")

        # Verify KV has updated data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None
        from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
            ConversationStatus,
        )
        kv_doc = ConversationStatus.model_validate_json(kv_value)
        # Compare timestamps (allow small differences due to serialization)
        assert abs((kv_doc.new_msg_start_time - new_time).total_seconds()) < 1
        logger.info("Step 3.5: KV-Storage updated successfully")

        # 4. Delete
        success = await repository.delete_by_group_id(test_group_id)
        assert success
        logger.info("Step 4: Deleted successfully")

        # 5. Verify deletion
        retrieved_after_delete = await repository.get_by_group_id(test_group_id)
        assert retrieved_after_delete is None
        kv_value_after_delete = await kv_storage.get(doc_id)
        assert kv_value_after_delete is None
        logger.info("Step 5: Verified deletion from both MongoDB and KV-Storage")

        logger.info("✅ Test passed: Full lifecycle completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
