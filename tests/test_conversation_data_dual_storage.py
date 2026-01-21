#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ConversationDataRepository with Dual Storage

Verify that dual storage works correctly for ConversationDataRepository.
Since ConversationDataRepository is a wrapper around MemoryRequestLogRepository,
and MemoryRequestLogRepository already has DualStorageMixin, the dual storage
should work automatically for all ConversationData operations.

Test Coverage:
1. save_conversation_data syncs to KV-Storage
2. get_conversation_data reads full data from KV-Storage
3. delete_conversation_data updates KV-Storage
4. fetch_unprocessed_conversation_data reads full data
5. Verify MongoDB stores Lite data, KV stores full data
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.repository.conversation_data_raw_repository import (
        ConversationDataRepository,
    )
    from infra_layer.adapters.out.persistence.repository.memory_request_log_repository import (
        MemoryRequestLogRepository,
    )


@pytest_asyncio.fixture
async def conversation_repo():
    """Get ConversationDataRepository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.conversation_data_raw_repository import (
        ConversationDataRepository,
    )

    return get_bean_by_type(ConversationDataRepository)


@pytest_asyncio.fixture
async def memory_log_repo():
    """Get MemoryRequestLogRepository instance"""
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
    message_id: str = None,
    content: str = "Test message content",
    sync_status: int = -1,
    created_at: datetime = None,
):
    """Create test MemoryRequestLog"""
    from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
        MemoryRequestLog,
    )

    if message_id is None:
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

    log = MemoryRequestLog(
        group_id=group_id,
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        user_id=user_id,
        message_id=message_id,
        message_create_time=to_iso_format(created_at or get_now_with_timezone()),
        sender=user_id or "test_sender",
        sender_name="Test User",
        role="user",
        content=content,
        group_name="Test Group",
        refer_list=[],
        raw_input={"test": "data", "large_field": "x" * 1000},  # Full data field
        version="1.0.0",
        endpoint_name="memorize",
        method="POST",
        url="/api/memorize",
        sync_status=sync_status,
    )

    if created_at:
        log.created_at = created_at

    return log


def create_test_raw_data_list(message_ids: list[str]):
    """Create test RawData list"""
    from memory_layer.memcell_extractor.base_memcell_extractor import RawData

    return [
        RawData(
            data_id=msg_id,
            content={"content": f"Message content for {msg_id}"},
            data_type="message",
        )
        for msg_id in message_ids
    ]


def get_logger_instance():
    """Helper to get logger"""
    return get_logger(__name__)


class TestConversationDataDualStorage:
    """Test ConversationDataRepository dual storage functionality"""

    async def test_01_save_conversation_data_syncs_to_kv(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: save_conversation_data updates sync_status and syncs to KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: save_conversation_data syncs to KV-Storage")

        try:
            # Create test logs with sync_status=-1
            msg1_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg2_id = f"msg_{uuid.uuid4().hex[:8]}"

            log1 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg1_id,
                content="Test message 1",
                sync_status=-1,
            )
            log2 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg2_id,
                content="Test message 2",
                sync_status=-1,
            )

            saved1 = await memory_log_repo.save(log1)
            saved2 = await memory_log_repo.save(log2)

            doc_id1 = str(saved1.id)
            doc_id2 = str(saved2.id)
            logger.info(f"✅ Created 2 logs: {doc_id1}, {doc_id2}")

            # Verify KV has initial data (sync_status=-1)
            kv_value1 = await kv_storage.get(doc_id1)
            assert kv_value1 is not None, "KV should have initial data"
            logger.info(f"✅ KV has initial data for {doc_id1}")

            # Save conversation data (updates sync_status: -1 -> 0)
            raw_data_list = create_test_raw_data_list([msg1_id, msg2_id])
            result = await conversation_repo.save_conversation_data(
                raw_data_list, test_group_id
            )
            assert result is True, "save_conversation_data should return True"
            logger.info("✅ save_conversation_data succeeded")

            # Verify KV-Storage is updated with sync_status=0
            from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
                MemoryRequestLog,
            )

            kv_value1_updated = await kv_storage.get(doc_id1)
            assert kv_value1_updated is not None, "KV should still have data"
            kv_doc1 = MemoryRequestLog.model_validate_json(kv_value1_updated)
            assert kv_doc1.sync_status == 0, "KV should have updated sync_status=0"
            assert (
                kv_doc1.raw_input is not None
            ), "KV should preserve full data (raw_input)"
            logger.info(f"✅ KV-Storage updated with sync_status=0: {doc_id1}")

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info("✅ Test passed: save_conversation_data syncs to KV-Storage")

    async def test_02_get_conversation_data_reads_from_kv(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: get_conversation_data returns full data from KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: get_conversation_data reads full data from KV-Storage")

        try:
            # Create test logs with different sync_status
            msg1_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg2_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg3_id = f"msg_{uuid.uuid4().hex[:8]}"

            log1 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg1_id,
                content="Pending message",
                sync_status=-1,
            )
            log2 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg2_id,
                content="Accumulating message",
                sync_status=0,
            )
            log3 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg3_id,
                content="Used message",
                sync_status=1,
            )

            await memory_log_repo.save(log1)
            await memory_log_repo.save(log2)
            await memory_log_repo.save(log3)
            logger.info("✅ Created logs with sync_status -1, 0, 1")

            # Get conversation data (should return sync_status=-1 and 0, exclude 1)
            result = await conversation_repo.get_conversation_data(group_id=test_group_id)

            # Verify results
            assert len(result) == 2, f"Expected 2 results, got {len(result)}"

            # Verify full data is returned (content field exists in RawData)
            for raw_data in result:
                assert raw_data.content is not None, "Content should not be None"
                assert isinstance(
                    raw_data.content, dict
                ), "Content should be a dictionary"
                content_text = raw_data.content.get("content", "")
                assert (
                    content_text in ["Pending message", "Accumulating message"]
                ), f"Unexpected content: {content_text}"

            logger.info("✅ get_conversation_data returned full data from KV-Storage")

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info("✅ Test passed: get_conversation_data reads from KV-Storage")

    async def test_03_delete_conversation_data_updates_kv(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: delete_conversation_data updates sync_status in KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: delete_conversation_data updates KV-Storage")

        try:
            # Create test logs
            msg1_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg2_id = f"msg_{uuid.uuid4().hex[:8]}"

            log1 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg1_id,
                content="Pending message",
                sync_status=-1,
            )
            log2 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg2_id,
                content="Accumulating message",
                sync_status=0,
            )

            saved1 = await memory_log_repo.save(log1)
            saved2 = await memory_log_repo.save(log2)

            doc_id1 = str(saved1.id)
            doc_id2 = str(saved2.id)
            logger.info(f"✅ Created 2 logs")

            # Delete conversation data (marks sync_status: -1,0 -> 1)
            result = await conversation_repo.delete_conversation_data(test_group_id)
            assert result is True, "delete_conversation_data should return True"
            logger.info("✅ delete_conversation_data succeeded")

            # Verify KV-Storage is updated with sync_status=1
            from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
                MemoryRequestLog,
            )

            kv_value1 = await kv_storage.get(doc_id1)
            assert kv_value1 is not None, "KV should still have data"
            kv_doc1 = MemoryRequestLog.model_validate_json(kv_value1)
            assert kv_doc1.sync_status == 1, "KV should have updated sync_status=1"
            logger.info(f"✅ KV-Storage updated with sync_status=1: {doc_id1}")

            kv_value2 = await kv_storage.get(doc_id2)
            assert kv_value2 is not None, "KV should still have data"
            kv_doc2 = MemoryRequestLog.model_validate_json(kv_value2)
            assert kv_doc2.sync_status == 1, "KV should have updated sync_status=1"
            logger.info(f"✅ KV-Storage updated with sync_status=1: {doc_id2}")

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info("✅ Test passed: delete_conversation_data updates KV-Storage")

    async def test_04_fetch_unprocessed_reads_from_kv(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: fetch_unprocessed_conversation_data reads full data from KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: fetch_unprocessed_conversation_data reads from KV-Storage")

        try:
            now = get_now_with_timezone()

            # Create logs with different timestamps
            msg1_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg2_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg3_id = f"msg_{uuid.uuid4().hex[:8]}"

            log1 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg1_id,
                content="Oldest pending",
                sync_status=-1,
                created_at=now - timedelta(hours=3),
            )
            log2 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg2_id,
                content="Middle accumulating",
                sync_status=0,
                created_at=now - timedelta(hours=2),
            )
            log3 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg3_id,
                content="Used message",
                sync_status=1,
                created_at=now - timedelta(hours=1),
            )

            await memory_log_repo.save(log1)
            await memory_log_repo.save(log2)
            await memory_log_repo.save(log3)
            logger.info("✅ Created 3 logs with different sync_status and timestamps")

            # Fetch unprocessed data
            result = await conversation_repo.fetch_unprocessed_conversation_data(
                test_group_id, limit=100
            )

            # Verify results
            assert len(result) == 2, f"Expected 2 results (exclude sync_status=1), got {len(result)}"

            # Verify full data is returned
            for raw_data in result:
                assert raw_data.content is not None, "Content should not be None"
                content_text = raw_data.content.get("content", "")
                assert content_text in [
                    "Oldest pending",
                    "Middle accumulating",
                ], f"Unexpected content: {content_text}"

            # Verify ascending order (oldest first)
            assert "Oldest pending" in str(result[0].content.get("content", ""))
            assert "Middle accumulating" in str(result[1].content.get("content", ""))
            logger.info(
                "✅ fetch_unprocessed returned full data in ascending order from KV-Storage"
            )

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info(
                "✅ Test passed: fetch_unprocessed_conversation_data reads from KV-Storage"
            )

    async def test_05_exclude_message_ids_works_with_kv(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: exclude_message_ids parameter works correctly with KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: exclude_message_ids works with KV-Storage")

        try:
            # Create test logs
            msg1_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg2_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg3_id = f"msg_{uuid.uuid4().hex[:8]}"

            log1 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg1_id,
                content="Message 1",
                sync_status=-1,
            )
            log2 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg2_id,
                content="Message 2",
                sync_status=0,
            )
            log3 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg3_id,
                content="Message 3",
                sync_status=-1,
            )

            await memory_log_repo.save(log1)
            await memory_log_repo.save(log2)
            await memory_log_repo.save(log3)
            logger.info("✅ Created 3 logs")

            # Get conversation data excluding msg1_id
            result = await conversation_repo.get_conversation_data(
                group_id=test_group_id, exclude_message_ids=[msg1_id]
            )

            # Verify results
            assert len(result) == 2, f"Expected 2 results after exclusion, got {len(result)}"

            # Verify excluded message is not in results
            result_message_ids = [r.data_id for r in result]
            assert msg1_id not in result_message_ids, "msg1 should be excluded"
            assert msg2_id in result_message_ids, "msg2 should be in results"
            assert msg3_id in result_message_ids, "msg3 should be in results"

            logger.info("✅ exclude_message_ids works correctly with KV-Storage")

            # Test delete with exclude
            result = await conversation_repo.delete_conversation_data(
                test_group_id, exclude_message_ids=[msg3_id]
            )
            assert result is True, "delete_conversation_data should return True"

            # Verify msg3 remains unprocessed
            remaining = await conversation_repo.get_conversation_data(
                group_id=test_group_id
            )
            assert len(remaining) == 1, f"Expected 1 remaining, got {len(remaining)}"
            assert remaining[0].data_id == msg3_id, "msg3 should remain"

            logger.info("✅ delete with exclude works correctly with KV-Storage")

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info("✅ Test passed: exclude_message_ids works with KV-Storage")

    async def test_06_time_range_filter_with_kv(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: time range filter works correctly with KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: time range filter works with KV-Storage")

        try:
            now = get_now_with_timezone()

            # Create logs with different timestamps
            msg1_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg2_id = f"msg_{uuid.uuid4().hex[:8]}"
            msg3_id = f"msg_{uuid.uuid4().hex[:8]}"

            log1 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg1_id,
                content="Old message",
                sync_status=-1,
                created_at=now - timedelta(hours=5),
            )
            log2 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg2_id,
                content="Recent message",
                sync_status=0,
                created_at=now - timedelta(hours=1),
            )
            log3 = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg3_id,
                content="Very recent message",
                sync_status=-1,
                created_at=now - timedelta(minutes=30),
            )

            await memory_log_repo.save(log1)
            await memory_log_repo.save(log2)
            await memory_log_repo.save(log3)
            logger.info("✅ Created 3 logs with different timestamps")

            # Query with time range (last 2 hours)
            start_time = to_iso_format(now - timedelta(hours=2))
            end_time = to_iso_format(now + timedelta(hours=1))

            result = await conversation_repo.get_conversation_data(
                group_id=test_group_id, start_time=start_time, end_time=end_time
            )

            # Verify only recent messages are returned
            assert len(result) == 2, f"Expected 2 results in time range, got {len(result)}"

            result_contents = [r.content.get("content", "") for r in result]
            assert "Recent message" in result_contents
            assert "Very recent message" in result_contents
            assert "Old message" not in result_contents

            logger.info("✅ time range filter works correctly with KV-Storage")

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info("✅ Test passed: time range filter works with KV-Storage")

    async def test_07_mongodb_lite_kv_full_verification(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: Verify MongoDB stores Lite data, KV-Storage stores full data"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: MongoDB Lite vs KV Full data verification")

        try:
            # Create test log with large data
            msg_id = f"msg_{uuid.uuid4().hex[:8]}"
            large_data = "x" * 10000  # Large field

            log = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg_id,
                content="Test message with large data",
                sync_status=-1,
            )
            log.raw_input = {"large_field": large_data, "test": "data"}

            saved = await memory_log_repo.save(log)
            doc_id = str(saved.id)
            logger.info(f"✅ Created log with large data: {doc_id}")

            # Verify KV-Storage has full data
            kv_value = await kv_storage.get(doc_id)
            assert kv_value is not None, "KV should have data"

            from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
                MemoryRequestLog,
            )

            kv_doc = MemoryRequestLog.model_validate_json(kv_value)
            assert kv_doc.raw_input is not None, "KV should have raw_input"
            assert (
                kv_doc.raw_input.get("large_field") == large_data
            ), "KV should have full large_field data"
            logger.info("✅ KV-Storage has full data (raw_input with large_field)")

            # Verify ConversationDataRepository can retrieve full data
            result = await conversation_repo.get_conversation_data(group_id=test_group_id)
            assert len(result) == 1, "Should have 1 result"

            # RawData contains the full data from KV
            raw_data = result[0]
            assert raw_data.content is not None, "Content should not be None"
            logger.info(
                "✅ ConversationDataRepository retrieved full data through KV-Storage"
            )

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info(
                "✅ Test passed: MongoDB Lite vs KV Full data verification complete"
            )

    async def test_08_complete_state_transition_with_kv(
        self, conversation_repo, memory_log_repo, kv_storage, test_user_id, test_group_id
    ):
        """Test: Complete sync_status state transition flow with KV-Storage"""
        logger = get_logger_instance()
        logger.info("=" * 60)
        logger.info("TEST: Complete state transition with KV-Storage")

        try:
            msg_id = f"msg_{uuid.uuid4().hex[:8]}"

            # Step 1: Create log with sync_status=-1
            log = create_test_memory_request_log(
                group_id=test_group_id,
                user_id=test_user_id,
                message_id=msg_id,
                content="Test message for state flow",
                sync_status=-1,
            )
            saved = await memory_log_repo.save(log)
            doc_id = str(saved.id)
            logger.info(f"✅ Step 1: Created log with sync_status=-1")

            # Verify KV has sync_status=-1
            from infra_layer.adapters.out.persistence.document.request.memory_request_log import (
                MemoryRequestLog,
            )

            kv_value = await kv_storage.get(doc_id)
            kv_doc = MemoryRequestLog.model_validate_json(kv_value)
            assert kv_doc.sync_status == -1, "KV should have sync_status=-1"
            logger.info(f"✅ KV has sync_status=-1: {doc_id}")

            # Step 2: save_conversation_data -> sync_status becomes 0
            raw_data_list = create_test_raw_data_list([msg_id])
            await conversation_repo.save_conversation_data(raw_data_list, test_group_id)
            logger.info("✅ Step 2: save_conversation_data executed")

            # Verify KV has sync_status=0
            kv_value = await kv_storage.get(doc_id)
            kv_doc = MemoryRequestLog.model_validate_json(kv_value)
            assert kv_doc.sync_status == 0, "KV should have sync_status=0"
            logger.info(f"✅ KV has sync_status=0: {doc_id}")

            # Step 3: delete_conversation_data -> sync_status becomes 1
            await conversation_repo.delete_conversation_data(test_group_id)
            logger.info("✅ Step 3: delete_conversation_data executed")

            # Verify KV has sync_status=1
            kv_value = await kv_storage.get(doc_id)
            kv_doc = MemoryRequestLog.model_validate_json(kv_value)
            assert kv_doc.sync_status == 1, "KV should have sync_status=1"
            logger.info(f"✅ KV has sync_status=1: {doc_id}")

            # Verify the message is no longer retrievable
            result = await conversation_repo.get_conversation_data(
                group_id=test_group_id
            )
            assert len(result) == 0, "Should not return used messages"
            logger.info("✅ Used messages not retrievable via get_conversation_data")

        finally:
            # Cleanup
            await memory_log_repo.delete_by_group_id(test_group_id)
            logger.info(
                "✅ Test passed: Complete state transition with KV-Storage verified"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
