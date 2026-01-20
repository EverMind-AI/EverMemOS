#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test MemoryRequestLogRepository with DualStorageMixin

Verify that DualStorageMixin works correctly for MemoryRequestLog.
Repository code remains unchanged, all dual storage logic is handled transparently by Mixin.
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from core.observation.logger import get_logger

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.memory_request_log_repository import (
        MemoryRequestLogRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.memory_request_log_repository import (
        MemoryRequestLogRepository,
    )
    return get_bean_by_type(MemoryRequestLogRepository)


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


@pytest.fixture
def test_group_id():
    """Generate unique test group ID"""
    return f"test_group_{uuid.uuid4().hex[:8]}"


def create_test_memory_request_log(
    group_id: str,
    user_id: str = None,
    request_id: str = None,
    message_id: str = None,
    content: str = "Test message content",
    sync_status: int = -1,
):
    """Create test MemoryRequestLog"""
    from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
        MemoryRequestLog,
    )

    if request_id is None:
        request_id = f"req_{uuid.uuid4().hex[:8]}"
    if message_id is None:
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

    return MemoryRequestLog(
        group_id=group_id,
        request_id=request_id,
        user_id=user_id,
        message_id=message_id,
        message_create_time=datetime.now().isoformat(),
        sender=user_id,
        sender_name="Test User",
        role="user",
        content=content,
        group_name="Test Group",
        refer_list=[],
        raw_input={"test": "data"},
        version="1.0.0",
        endpoint_name="memorize",
        method="POST",
        url="/api/memorize",
        sync_status=sync_status,
    )


@pytest.mark.asyncio
class TestMemoryRequestLogDualStorage:
    """Test MemoryRequestLog dual storage functionality"""

    async def test_01_save_syncs_to_kv(self, repository, kv_storage, test_user_id, test_group_id):
        """Test: save() syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog save() syncs to KV-Storage")

        # Create test data
        test_data = create_test_memory_request_log(
            group_id=test_group_id,
            user_id=test_user_id,
            content="Test save to KV"
        )
        saved = await repository.save(test_data)

        assert saved is not None, "save failed"
        doc_id = str(saved.id)
        logger.info(f"✅ Saved: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"

        # Verify full data in KV
        from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
            MemoryRequestLog,
        )

        kv_doc = MemoryRequestLog.model_validate_json(kv_value)
        assert kv_doc.content == "Test save to KV"
        assert kv_doc.user_id == test_user_id
        assert kv_doc.group_id == test_group_id
        logger.info("✅ Test passed: save() syncs to KV-Storage")

    async def test_02_get_by_request_id_reads_from_kv(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: get_by_request_id() reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog get_by_request_id() reads from KV-Storage")

        # Create test data
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        test_data = create_test_memory_request_log(
            group_id=test_group_id,
            user_id=test_user_id,
            request_id=request_id,
            content="Test get by request_id"
        )
        saved = await repository.save(test_data)
        assert saved is not None
        logger.info(f"✅ Created: request_id={request_id}")

        # Get by request_id
        retrieved = await repository.get_by_request_id(request_id)
        assert retrieved is not None, "get_by_request_id failed"
        assert retrieved.content == "Test get by request_id"
        assert retrieved.user_id == test_user_id
        assert retrieved.group_id == test_group_id
        logger.info("✅ Test passed: get_by_request_id() reads from KV-Storage")

    async def test_03_find_by_group_id_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: find_by_group_id() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog find_by_group_id() works with dual storage")

        # Create multiple test records
        for i in range(3):
            test_data = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                content=f"Test message {i+1}",
                sync_status=0  # In window accumulation
            )
            await repository.save(test_data)

        # Query by group_id
        results = await repository.find_by_group_id(
            group_id=test_group_id,
            sync_status=0,
            limit=10
        )
        assert len(results) == 3, f"Should return 3 records, got {len(results)}"

        # Verify full data
        for result in results:
            assert result.content is not None
            assert result.user_id == test_user_id
            assert result.group_id == test_group_id
            assert result.raw_input is not None  # Full data field

        logger.info("✅ Test passed: find_by_group_id() returns full data")

    async def test_04_find_by_group_id_with_statuses_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: find_by_group_id_with_statuses() returns full data"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog find_by_group_id_with_statuses() works")

        # Create records with different sync_status
        for status in [-1, 0, 1]:
            for i in range(2):
                test_data = create_test_memory_request_log(
                    group_id=test_group_id,
                    user_id=test_user_id,
                    content=f"Test status={status}, {i+1}",
                    sync_status=status
                )
                await repository.save(test_data)

        # Query for multiple statuses
        results = await repository.find_by_group_id_with_statuses(
            group_id=test_group_id,
            sync_status_list=[-1, 0],
            limit=10
        )
        assert len(results) == 4, f"Should return 4 records (2x -1 + 2x 0), got {len(results)}"

        # Verify all have the right status
        for result in results:
            assert result.sync_status in [-1, 0]
            assert result.content is not None

        logger.info("✅ Test passed: find_by_group_id_with_statuses() works")

    async def test_05_find_by_user_id_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: find_by_user_id() returns full data"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog find_by_user_id() works")

        # Create multiple records
        for i in range(3):
            test_data = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                content=f"User message {i+1}"
            )
            await repository.save(test_data)

        # Query by user_id
        results = await repository.find_by_user_id(user_id=test_user_id, limit=10)
        assert len(results) >= 3, f"Should return at least 3 records, got {len(results)}"

        # Verify full data
        for result in results[:3]:
            assert result.user_id == test_user_id
            assert result.content is not None

        logger.info("✅ Test passed: find_by_user_id() works")

    async def test_06_delete_by_group_id_removes_from_kv(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: delete_by_group_id() removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog delete_by_group_id() removes from KV")

        # Create test records
        doc_ids = []
        for i in range(3):
            test_data = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                content=f"To be deleted {i+1}"
            )
            saved = await repository.save(test_data)
            doc_ids.append(str(saved.id))

        # Verify KV has all data
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None

        # Delete by group_id
        deleted_count = await repository.delete_by_group_id(test_group_id)
        assert deleted_count == 3, f"Should delete 3 records, got {deleted_count}"

        # Verify KV removed all
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is None, f"KV-Storage should not have {doc_id} after delete"

        logger.info("✅ Test passed: delete_by_group_id() removes from KV")

    async def test_07_confirm_accumulation_by_group_id_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: confirm_accumulation_by_group_id() updates sync_status and KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog confirm_accumulation_by_group_id() works")

        # Create records with sync_status=-1
        for i in range(3):
            test_data = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                content=f"Pending message {i+1}",
                sync_status=-1
            )
            await repository.save(test_data)

        # Confirm accumulation
        updated_count = await repository.confirm_accumulation_by_group_id(test_group_id)
        assert updated_count == 3, f"Should update 3 records, got {updated_count}"

        # Query updated records
        results = await repository.find_by_group_id(
            group_id=test_group_id,
            sync_status=0,
            limit=10
        )
        assert len(results) == 3, "All records should now have sync_status=0"

        # Verify KV also updated (full data should reflect sync_status=0)
        for result in results:
            kv_value = await kv_storage.get(str(result.id))
            assert kv_value is not None
            from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
                MemoryRequestLog,
            )
            kv_doc = MemoryRequestLog.model_validate_json(kv_value)
            assert kv_doc.sync_status == 0, "KV should have updated sync_status"

        logger.info("✅ Test passed: confirm_accumulation_by_group_id() works")

    async def test_08_mark_as_used_by_group_id_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: mark_as_used_by_group_id() updates sync_status to 1"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog mark_as_used_by_group_id() works")

        # Create records with sync_status=-1 and 0
        for status in [-1, 0]:
            for i in range(2):
                test_data = create_test_memory_request_log(
                    group_id=test_group_id,
                    user_id=test_user_id,
                    content=f"Status {status}, {i+1}",
                    sync_status=status
                )
                await repository.save(test_data)

        # Mark as used
        updated_count = await repository.mark_as_used_by_group_id(test_group_id)
        assert updated_count == 4, f"Should update 4 records (2x -1 + 2x 0), got {updated_count}"

        # Query for sync_status=1
        results = await repository.find_by_group_id(
            group_id=test_group_id,
            sync_status=1,
            limit=10
        )
        assert len(results) == 4, "All records should now have sync_status=1"

        # Verify KV updated
        for result in results:
            kv_value = await kv_storage.get(str(result.id))
            assert kv_value is not None
            from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
                MemoryRequestLog,
            )
            kv_doc = MemoryRequestLog.model_validate_json(kv_value)
            assert kv_doc.sync_status == 1, "KV should have updated sync_status"

        logger.info("✅ Test passed: mark_as_used_by_group_id() works")

    async def test_09_find_pending_by_filters_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: find_pending_by_filters() with MAGIC_ALL support"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog find_pending_by_filters() works")

        from core.oxm.constants import MAGIC_ALL

        # Create test records
        for i in range(3):
            test_data = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                content=f"Pending message {i+1}",
                sync_status=-1
            )
            await repository.save(test_data)

        # Query with user_id filter
        results = await repository.find_pending_by_filters(
            user_id=test_user_id,
            group_id=MAGIC_ALL,
            sync_status_list=[-1],
            limit=10
        )
        assert len(results) >= 3, f"Should return at least 3 records, got {len(results)}"

        # Query with group_id filter
        results = await repository.find_pending_by_filters(
            user_id=MAGIC_ALL,
            group_id=test_group_id,
            sync_status_list=[-1],
            limit=10
        )
        assert len(results) == 3, f"Should return 3 records, got {len(results)}"

        # Verify full data
        for result in results:
            assert result.content is not None
            assert result.group_id == test_group_id

        logger.info("✅ Test passed: find_pending_by_filters() works")

    async def test_10_confirm_accumulation_by_message_ids_works(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: confirm_accumulation_by_message_ids() updates specific records"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: MemoryRequestLog confirm_accumulation_by_message_ids() works")

        # Create records
        message_ids = []
        for i in range(3):
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
            message_ids.append(message_id)
            test_data = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=message_id,
                content=f"Message {i+1}",
                sync_status=-1
            )
            await repository.save(test_data)

        # Confirm only first 2 messages
        updated_count = await repository.confirm_accumulation_by_message_ids(
            group_id=test_group_id,
            message_ids=message_ids[:2]
        )
        assert updated_count == 2, f"Should update 2 records, got {updated_count}"

        # Verify: 2 records with sync_status=0, 1 record with sync_status=-1
        results_0 = await repository.find_by_group_id(
            group_id=test_group_id,
            sync_status=0,
            limit=10
        )
        assert len(results_0) == 2, "Should have 2 records with sync_status=0"

        results_minus1 = await repository.find_by_group_id(
            group_id=test_group_id,
            sync_status=-1,
            limit=10
        )
        assert len(results_minus1) == 1, "Should have 1 record with sync_status=-1"

        logger.info("✅ Test passed: confirm_accumulation_by_message_ids() works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
