#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test EventLogRecordRawRepository with DualStorageMixin

Verify that DualStorageMixin works correctly for EventLogRecord.
Repository code remains unchanged, all dual storage logic is handled transparently by Mixin.
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from core.observation.logger import get_logger
from api_specs.memory_types import ParentType

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
        EventLogRecordRawRepository,
    )


@pytest_asyncio.fixture
async def repository():
    """Get repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
        EventLogRecordRawRepository,
    )
    return get_bean_by_type(EventLogRecordRawRepository)


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


def create_test_event_log(
    user_id: str,
    atomic_fact: str = "Test atomic fact",
    group_id: str = None,
    parent_type: str = ParentType.MEMCELL.value,
    parent_id: str = None,
    timestamp: datetime = None,
):
    """Create test EventLogRecord"""
    from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
        EventLogRecord,
    )

    if timestamp is None:
        timestamp = datetime.now()
    if parent_id is None:
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"

    return EventLogRecord(
        user_id=user_id,
        user_name="Test User",
        group_id=group_id,
        atomic_fact=atomic_fact,
        parent_type=parent_type,
        parent_id=parent_id,
        timestamp=timestamp,
        participants=["Alice", "Bob"],
        event_type="Conversation",
        extend={"test_key": "test_value"},
    )


@pytest.mark.asyncio
class TestEventLogRecordDualStorage:
    """Test EventLogRecord dual storage functionality"""

    async def test_01_save_syncs_to_kv(self, repository, kv_storage, test_user_id):
        """Test: save() syncs to KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord save() syncs to KV-Storage")

        # Create test data
        test_data = create_test_event_log(user_id=test_user_id)
        saved = await repository.save(test_data)

        assert saved is not None, "save failed"
        doc_id = str(saved.id)
        logger.info(f"✅ Saved: {doc_id}")

        # Verify KV-Storage has the data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None, "KV-Storage should have the data"

        # Verify full data in KV
        from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
            EventLogRecord,
        )

        kv_doc = EventLogRecord.model_validate_json(kv_value)
        assert kv_doc.atomic_fact == test_data.atomic_fact
        assert kv_doc.user_id == test_user_id
        logger.info("✅ Test passed: save() syncs to KV-Storage")

    async def test_02_get_by_id_reads_from_kv(
        self, repository, kv_storage, test_user_id
    ):
        """Test: get_by_id() reads from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord get_by_id() reads from KV-Storage")

        # Create test data
        test_data = create_test_event_log(
            user_id=test_user_id, atomic_fact="Test get from KV"
        )
        saved = await repository.save(test_data)
        assert saved is not None
        doc_id = str(saved.id)
        logger.info(f"✅ Created: {doc_id}")

        # Get by ID
        retrieved = await repository.get_by_id(doc_id)
        assert retrieved is not None, "get_by_id failed"
        assert retrieved.atomic_fact == "Test get from KV"
        assert retrieved.user_id == test_user_id
        logger.info("✅ Test passed: get_by_id() reads from KV-Storage")

    async def test_03_find_by_filters_works(
        self, repository, kv_storage, test_user_id
    ):
        """Test: find_by_filters() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord find_by_filters() works with dual storage")

        # Create multiple test records
        for i in range(3):
            test_data = create_test_event_log(
                user_id=test_user_id, atomic_fact=f"Test fact {i+1}"
            )
            await repository.save(test_data)

        # Query by user_id
        results = await repository.find_by_filters(user_id=test_user_id, limit=10)
        assert len(results) == 3, f"Should return 3 records, got {len(results)}"

        # Verify full data
        for result in results:
            assert result.atomic_fact is not None
            assert result.user_id == test_user_id
            assert result.extend is not None  # Full data field

        logger.info("✅ Test passed: find_by_filters() returns full data")

    async def test_04_delete_by_id_removes_from_kv(
        self, repository, kv_storage, test_user_id
    ):
        """Test: delete_by_id() removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord delete_by_id() removes from KV-Storage")

        # Create test data
        test_data = create_test_event_log(user_id=test_user_id)
        saved = await repository.save(test_data)
        doc_id = str(saved.id)

        # Verify KV has data
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is not None

        # Delete
        success = await repository.delete_by_id(doc_id)
        assert success, "delete_by_id should return True"

        # Verify KV removed
        kv_value = await kv_storage.get(doc_id)
        assert kv_value is None, "KV-Storage should not have the data after delete"

        logger.info("✅ Test passed: delete_by_id() removes from KV-Storage")

    async def test_05_get_by_parent_id_works(
        self, repository, kv_storage, test_user_id
    ):
        """Test: get_by_parent_id() returns full data from KV"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord get_by_parent_id() works")

        parent_id = f"parent_{uuid.uuid4().hex[:8]}"

        # Create multiple records with same parent_id
        for i in range(3):
            test_data = create_test_event_log(
                user_id=test_user_id,
                atomic_fact=f"Test fact {i+1}",
                parent_id=parent_id,
            )
            await repository.save(test_data)

        # Query by parent_id
        results = await repository.get_by_parent_id(parent_id)
        assert len(results) == 3, f"Should return 3 records, got {len(results)}"

        # Verify full data
        for result in results:
            assert result.parent_id == parent_id
            assert result.atomic_fact is not None

        logger.info("✅ Test passed: get_by_parent_id() returns full data")

    async def test_06_delete_by_parent_id_removes_from_kv(
        self, repository, kv_storage, test_user_id
    ):
        """Test: delete_by_parent_id() removes from KV-Storage"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord delete_by_parent_id() removes from KV")

        parent_id = f"parent_{uuid.uuid4().hex[:8]}"

        # Create multiple records with same parent_id
        doc_ids = []
        for i in range(3):
            test_data = create_test_event_log(
                user_id=test_user_id, parent_id=parent_id
            )
            saved = await repository.save(test_data)
            doc_ids.append(str(saved.id))

        # Verify KV has all data
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None

        # Delete by parent_id
        deleted_count = await repository.delete_by_parent_id(parent_id)
        assert deleted_count == 3, f"Should delete 3 records, got {deleted_count}"

        # Verify KV removed all
        for doc_id in doc_ids:
            kv_value = await kv_storage.get(doc_id)
            assert (
                kv_value is None
            ), f"KV-Storage should not have {doc_id} after delete"

        logger.info("✅ Test passed: delete_by_parent_id() removes from KV")

    async def test_07_find_with_time_range(
        self, repository, kv_storage, test_user_id
    ):
        """Test: find_by_filters() with time range works correctly"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord find_by_filters() with time range")

        now = datetime.now()

        # Create records with different timestamps
        timestamps = [
            now - timedelta(days=2),
            now - timedelta(days=1),
            now,
        ]

        for i, ts in enumerate(timestamps):
            test_data = create_test_event_log(
                user_id=test_user_id,
                atomic_fact=f"Test fact {i+1}",
                timestamp=ts,
            )
            await repository.save(test_data)

        # Query with time range (last 36 hours)
        start_time = now - timedelta(hours=36)
        results = await repository.find_by_filters(
            user_id=test_user_id, start_time=start_time
        )

        assert len(results) == 2, f"Should return 2 records, got {len(results)}"

        logger.info("✅ Test passed: find_by_filters() with time range works")

    async def test_08_find_with_group_filter(
        self, repository, kv_storage, test_user_id, test_group_id
    ):
        """Test: find_by_filters() with group_id filter"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord find_by_filters() with group_id")

        # Create records with and without group_id
        for i in range(2):
            test_data = create_test_event_log(
                user_id=test_user_id,
                group_id=test_group_id,
                atomic_fact=f"Group fact {i+1}",
            )
            await repository.save(test_data)

        for i in range(2):
            test_data = create_test_event_log(
                user_id=test_user_id, group_id=None, atomic_fact=f"Personal fact {i+1}"
            )
            await repository.save(test_data)

        # Query by group_id
        results = await repository.find_by_filters(
            user_id=test_user_id, group_id=test_group_id
        )
        assert len(results) == 2, f"Should return 2 group records, got {len(results)}"

        for result in results:
            assert result.group_id == test_group_id

        logger.info("✅ Test passed: find_by_filters() with group_id works")

    async def test_09_pagination_works(self, repository, kv_storage, test_user_id):
        """Test: find_by_filters() with limit and skip"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("TEST: EventLogRecord pagination (limit and skip)")

        # Create 10 records
        for i in range(10):
            test_data = create_test_event_log(
                user_id=test_user_id, atomic_fact=f"Test fact {i+1}"
            )
            await repository.save(test_data)

        # Get first page (5 records)
        page1 = await repository.find_by_filters(user_id=test_user_id, limit=5)
        assert len(page1) == 5, f"First page should have 5 records, got {len(page1)}"

        # Get second page (skip 5, limit 5)
        page2 = await repository.find_by_filters(
            user_id=test_user_id, skip=5, limit=5
        )
        assert len(page2) == 5, f"Second page should have 5 records, got {len(page2)}"

        # Verify no overlap
        page1_ids = {str(r.id) for r in page1}
        page2_ids = {str(r.id) for r in page2}
        assert page1_ids.isdisjoint(page2_ids), "Pages should not overlap"

        logger.info("✅ Test passed: Pagination works correctly")

    async def test_10_projection_model_not_supported(
        self, repository, kv_storage, test_user_id
    ):
        """Test: Projection model is not supported in Lite storage mode"""
        logger = get_logger()
        logger.info("=" * 60)
        logger.info(
            "TEST: EventLogRecord projection model (not supported in Lite mode)"
        )

        # Create test data
        test_data = create_test_event_log(user_id=test_user_id)
        saved = await repository.save(test_data)
        doc_id = str(saved.id)

        from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
            EventLogRecordProjection,
        )

        # Note: In Lite storage mode, projection is not supported
        # because we always load full data from KV-Storage
        # The projection_model parameter is ignored
        result = await repository.get_by_id(doc_id, model=EventLogRecordProjection)

        # Result should be full EventLogRecord (not projection)
        assert result is not None
        # In dual storage mode, we always get full data from KV
        assert hasattr(result, "vector")  # Full model has vector field

        logger.info(
            "ℹ️  Note: Projection models are not supported in Lite storage mode"
        )
        logger.info("✅ Test passed: Handles projection model gracefully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
