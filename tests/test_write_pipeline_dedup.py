"""
Tests for write pipeline deduplication and foresight expiry cleanup.

Covers:
- save_memory_docs: deletes old episodic records before inserting new ones
- save_memory_docs: deletes old event_log records before inserting new ones
- cleanup_expired_foresights: removes expired foresight records
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from collections import defaultdict
from typing import List, Dict, Any

from api_specs.memory_models import MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episodic_doc(parent_id: str = "memcell_001", episode: str = "test episode"):
    """Create a mock episodic memory document."""
    doc = MagicMock()
    doc.parent_id = parent_id
    doc.episode = episode
    doc.vector = [0.1, 0.2, 0.3]
    doc.id = "ep_001"
    doc.event_id = "ep_001"
    return doc


def _make_event_log_doc(parent_id: str = "memcell_001"):
    """Create a mock event log document."""
    doc = MagicMock()
    doc.parent_id = parent_id
    doc.atomic_fact = "test fact"
    doc.id = "el_001"
    return doc


def _make_foresight_doc(parent_id: str = "memcell_001"):
    """Create a mock foresight document."""
    doc = MagicMock()
    doc.parent_id = parent_id
    doc.content = "test foresight"
    doc.id = "fs_001"
    return doc


# ---------------------------------------------------------------------------
# Test: Episodic dedup in save_memory_docs
# ---------------------------------------------------------------------------

class TestEpisodicDedup:
    """Verify that save_memory_docs deletes old episodic records before insert."""

    @pytest.mark.asyncio
    async def test_dedup_deletes_old_records_before_insert(self):
        """
        When saving episodic docs, old records with the same parent_id
        should be deleted from MongoDB, ES, and Milvus before new insert.
        """
        # Arrange
        mock_episodic_repo = AsyncMock()
        mock_episodic_repo.append_episodic_memory = AsyncMock(
            side_effect=lambda doc: doc
        )
        mock_episodic_repo.delete_by_parent_id = AsyncMock(return_value=1)

        mock_es_repo = AsyncMock()
        mock_es_repo.create = AsyncMock()
        mock_es_repo.delete_by_filters = AsyncMock()

        mock_milvus_repo = AsyncMock()
        mock_milvus_repo.insert = AsyncMock()
        mock_milvus_repo.delete_by_filters = AsyncMock()

        doc = _make_episodic_doc(parent_id="mc_123")
        from biz_layer.mem_memorize import MemoryDocPayload

        payloads = [MemoryDocPayload(MemoryType.EPISODIC_MEMORY, doc)]

        with patch("biz_layer.mem_memorize.get_bean_by_type") as mock_get_bean, \
             patch("biz_layer.mem_memorize.EpisodicMemoryConverter") as mock_converter, \
             patch("biz_layer.mem_memorize.EpisodicMemoryMilvusConverter") as mock_milvus_converter:

            def _bean_router(cls):
                from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
                    EpisodicMemoryRawRepository,
                )
                from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
                    EpisodicMemoryEsRepository,
                )
                from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
                    EpisodicMemoryMilvusRepository,
                )

                repo_map = {
                    EpisodicMemoryRawRepository: mock_episodic_repo,
                    EpisodicMemoryEsRepository: mock_es_repo,
                    EpisodicMemoryMilvusRepository: mock_milvus_repo,
                }
                return repo_map.get(cls, AsyncMock())

            mock_get_bean.side_effect = _bean_router
            mock_converter.from_mongo.return_value = MagicMock()
            mock_milvus_converter.from_mongo.return_value = {"vector": [0.1]}

            # Act
            from biz_layer.mem_memorize import save_memory_docs
            result = await save_memory_docs(payloads)

            # Assert: delete was called before insert
            mock_episodic_repo.delete_by_parent_id.assert_called_once_with("mc_123")
            mock_es_repo.delete_by_filters.assert_called_once_with(
                filters={"parent_id": "mc_123"}
            )
            mock_milvus_repo.delete_by_filters.assert_called_once_with(
                filters={"parent_id": "mc_123"}
            )
            # And insert still happened
            mock_episodic_repo.append_episodic_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_dedup_failure_does_not_block_insert(self):
        """If dedup delete fails, insert should still proceed."""
        mock_episodic_repo = AsyncMock()
        mock_episodic_repo.append_episodic_memory = AsyncMock(
            side_effect=lambda doc: doc
        )
        mock_episodic_repo.delete_by_parent_id = AsyncMock(
            side_effect=Exception("DB error")
        )

        mock_es_repo = AsyncMock()
        mock_milvus_repo = AsyncMock()

        doc = _make_episodic_doc(parent_id="mc_fail")
        from biz_layer.mem_memorize import MemoryDocPayload

        payloads = [MemoryDocPayload(MemoryType.EPISODIC_MEMORY, doc)]

        with patch("biz_layer.mem_memorize.get_bean_by_type") as mock_get_bean, \
             patch("biz_layer.mem_memorize.EpisodicMemoryConverter") as mock_converter, \
             patch("biz_layer.mem_memorize.EpisodicMemoryMilvusConverter") as mock_milvus_converter:

            def _bean_router(cls):
                from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import (
                    EpisodicMemoryRawRepository,
                )
                from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
                    EpisodicMemoryEsRepository,
                )
                from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
                    EpisodicMemoryMilvusRepository,
                )
                repo_map = {
                    EpisodicMemoryRawRepository: mock_episodic_repo,
                    EpisodicMemoryEsRepository: mock_es_repo,
                    EpisodicMemoryMilvusRepository: mock_milvus_repo,
                }
                return repo_map.get(cls, AsyncMock())

            mock_get_bean.side_effect = _bean_router
            mock_converter.from_mongo.return_value = MagicMock()
            mock_milvus_converter.from_mongo.return_value = {"vector": [0.1]}

            from biz_layer.mem_memorize import save_memory_docs
            result = await save_memory_docs(payloads)

            # Insert still happened despite dedup failure
            mock_episodic_repo.append_episodic_memory.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Event log dedup in save_memory_docs
# ---------------------------------------------------------------------------

class TestEventLogDedup:
    """Verify that save_memory_docs deletes old event_log records before insert."""

    @pytest.mark.asyncio
    async def test_dedup_deletes_old_event_logs(self):
        mock_event_log_repo = AsyncMock()
        mock_event_log_repo.create_batch = AsyncMock(return_value=[])
        mock_event_log_repo.delete_by_parent_id = AsyncMock(return_value=2)

        mock_milvus_repo = AsyncMock()
        mock_milvus_repo.delete_by_parent_id = AsyncMock(return_value=True)

        mock_sync_service = AsyncMock()

        doc = _make_event_log_doc(parent_id="mc_el_001")
        from biz_layer.mem_memorize import MemoryDocPayload

        payloads = [MemoryDocPayload(MemoryType.EVENT_LOG, doc)]

        with patch("biz_layer.mem_memorize.get_bean_by_type") as mock_get_bean:
            def _bean_router(cls):
                from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import (
                    EventLogRecordRawRepository,
                )
                from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
                    EventLogMilvusRepository,
                )
                from biz_layer.mem_sync import MemorySyncService

                repo_map = {
                    EventLogRecordRawRepository: mock_event_log_repo,
                    EventLogMilvusRepository: mock_milvus_repo,
                    MemorySyncService: mock_sync_service,
                }
                return repo_map.get(cls, AsyncMock())

            mock_get_bean.side_effect = _bean_router

            from biz_layer.mem_memorize import save_memory_docs
            await save_memory_docs(payloads)

            # Verify delete was called
            mock_event_log_repo.delete_by_parent_id.assert_called_once_with("mc_el_001")
            mock_milvus_repo.delete_by_parent_id.assert_called_once_with("mc_el_001")
            # And batch create still happened
            mock_event_log_repo.create_batch.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Foresight expiry cleanup
# ---------------------------------------------------------------------------

class TestForesightCleanup:
    """Verify cleanup_expired_foresights removes expired records."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_records(self):
        """Expired foresight records should be deleted from all stores."""
        mock_record_1 = MagicMock()
        mock_record_1.id = "fs_expired_001"
        mock_record_2 = MagicMock()
        mock_record_2.id = "fs_expired_002"

        mock_foresight_repo = AsyncMock()
        mock_foresight_repo.delete_by_id = AsyncMock(return_value=True)

        mock_es_repo = AsyncMock()
        mock_es_repo.delete_by_filters = AsyncMock()

        mock_milvus_repo = AsyncMock()
        mock_milvus_repo.delete_by_id = AsyncMock(return_value=True)

        with patch(
            "biz_layer.mem_memorize.ForesightRecord"
        ) as MockForesightRecord, patch(
            "biz_layer.mem_memorize.get_bean_by_type"
        ) as mock_get_bean, patch(
            "biz_layer.mem_memorize.get_now_with_timezone"
        ) as mock_now, patch(
            "biz_layer.mem_memorize.to_date_str"
        ) as mock_to_date_str:

            # Setup: 2 expired records
            mock_find = MagicMock()
            mock_find.to_list = AsyncMock(
                return_value=[mock_record_1, mock_record_2]
            )
            MockForesightRecord.find.return_value = mock_find

            mock_now.return_value = datetime(2026, 3, 2)
            mock_to_date_str.return_value = "2026-03-02"

            def _bean_router(cls):
                from infra_layer.adapters.out.persistence.repository.foresight_record_repository import (
                    ForesightRecordRawRepository,
                )
                from infra_layer.adapters.out.search.repository.foresight_es_repository import (
                    ForesightEsRepository,
                )
                from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
                    ForesightMilvusRepository,
                )
                repo_map = {
                    ForesightRecordRawRepository: mock_foresight_repo,
                    ForesightEsRepository: mock_es_repo,
                    ForesightMilvusRepository: mock_milvus_repo,
                }
                return repo_map.get(cls, AsyncMock())

            mock_get_bean.side_effect = _bean_router

            from biz_layer.mem_memorize import cleanup_expired_foresights
            count = await cleanup_expired_foresights()

            # Should have deleted 2 records
            assert count == 2
            assert mock_foresight_repo.delete_by_id.call_count == 2
            assert mock_milvus_repo.delete_by_id.call_count == 2
            assert mock_es_repo.delete_by_filters.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_when_no_expired(self):
        """When there are no expired records, cleanup should return 0."""
        with patch(
            "biz_layer.mem_memorize.ForesightRecord"
        ) as MockForesightRecord, patch(
            "biz_layer.mem_memorize.get_now_with_timezone"
        ) as mock_now, patch(
            "biz_layer.mem_memorize.to_date_str"
        ) as mock_to_date_str:

            mock_find = MagicMock()
            mock_find.to_list = AsyncMock(return_value=[])
            MockForesightRecord.find.return_value = mock_find
            mock_now.return_value = datetime(2026, 3, 2)
            mock_to_date_str.return_value = "2026-03-02"

            from biz_layer.mem_memorize import cleanup_expired_foresights
            count = await cleanup_expired_foresights()

            assert count == 0
