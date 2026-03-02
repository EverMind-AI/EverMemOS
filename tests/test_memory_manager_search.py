"""Unit tests for multi-memory-type search logic in MemoryManager.

Tests cover:
- get_keyword_search_results iterating over multiple memory types
- get_vector_search_results iterating over multiple memory types
- Unsupported types (e.g. PROFILE) silently skipped
- Single memory type still works
- Empty memory_types list returns empty
- _memory_types_label helper
"""

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Ensure src/ is on the path so that imports like `api_specs.*` resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class _FakeEmbedding:
    """Mimics an np.ndarray with .tolist() — avoids numpy import at test time."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)

from api_specs.memory_models import MemoryType, RetrieveMethod
from api_specs.dtos import RetrieveMemRequest


# ---------------------------------------------------------------------------
# Helper: build a RetrieveMemRequest without triggering validation side-effects
# ---------------------------------------------------------------------------
def _make_request(
    memory_types,
    query="test query",
    user_id="u1",
    group_id="g1",
    top_k=10,
):
    return RetrieveMemRequest(
        user_id=user_id,
        group_id=group_id,
        memory_types=memory_types,
        query=query,
        top_k=top_k,
        retrieve_method=RetrieveMethod.KEYWORD,
    )


# ---------------------------------------------------------------------------
# 7. _memory_types_label
# ---------------------------------------------------------------------------
class TestMemoryTypesLabel:
    def test_single_type(self):
        from agentic_layer.memory_manager import _memory_types_label

        assert _memory_types_label([MemoryType.EPISODIC_MEMORY]) == "episodic_memory"

    def test_multiple_types(self):
        from agentic_layer.memory_manager import _memory_types_label

        result = _memory_types_label(
            [MemoryType.EPISODIC_MEMORY, MemoryType.FORESIGHT]
        )
        assert result == "episodic_memory,foresight"

    def test_empty_list(self):
        from agentic_layer.memory_manager import _memory_types_label

        assert _memory_types_label([]) == "unknown"

    def test_three_types(self):
        from agentic_layer.memory_manager import _memory_types_label

        result = _memory_types_label(
            [MemoryType.EPISODIC_MEMORY, MemoryType.FORESIGHT, MemoryType.EVENT_LOG]
        )
        assert result == "episodic_memory,foresight,event_log"


# ---------------------------------------------------------------------------
# Fixtures shared by keyword / vector tests
# ---------------------------------------------------------------------------

# Patch targets – all within the memory_manager module
_MM = "agentic_layer.memory_manager"


def _patch_constructor():
    """Patch MemoryManager.__init__ so it doesn't need real DI beans."""
    return patch(f"{_MM}.MemoryManager.__init__", lambda self: None)


@pytest.fixture
def manager():
    """Return a MemoryManager instance with __init__ patched out."""
    with _patch_constructor():
        from agentic_layer.memory_manager import MemoryManager

        return MemoryManager()


# ---------------------------------------------------------------------------
# 1 & 2 & 5 & 6. get_keyword_search_results
# ---------------------------------------------------------------------------
class TestGetKeywordSearchResults:

    @pytest.mark.asyncio
    async def test_multiple_memory_types_merged(self, manager):
        """Multiple memory types → results from all repos are merged."""
        episodic_hits = [{"_id": "e1", "_score": 1.0, "summary": "ep hit"}]
        foresight_hits = [{"_id": "f1", "_score": 0.8, "content": "fore hit"}]

        mock_ep_repo = MagicMock()
        mock_ep_repo.multi_search = AsyncMock(return_value=episodic_hits)
        mock_fore_repo = MagicMock()
        mock_fore_repo.multi_search = AsyncMock(return_value=foresight_hits)

        def _bean_router(cls):
            from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
                EpisodicMemoryEsRepository,
            )
            from infra_layer.adapters.out.search.repository.foresight_es_repository import (
                ForesightEsRepository,
            )

            if cls is EpisodicMemoryEsRepository:
                return mock_ep_repo
            if cls is ForesightEsRepository:
                return mock_fore_repo
            raise ValueError(f"Unexpected class: {cls}")

        req = _make_request(
            memory_types=[MemoryType.EPISODIC_MEMORY, MemoryType.FORESIGHT],
        )

        with (
            patch(f"{_MM}.get_bean_by_type", side_effect=_bean_router),
            patch(f"{_MM}.jieba") as mock_jieba,
            patch(f"{_MM}.filter_stopwords", return_value=["test", "query"]),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            mock_jieba.cut_for_search.return_value = ["test", "query"]
            results = await manager.get_keyword_search_results(req)

        # Both repos were called
        mock_ep_repo.multi_search.assert_awaited_once()
        mock_fore_repo.multi_search.assert_awaited_once()

        # Results merged
        assert len(results) == 2
        # Each hit annotated with its memory_type
        types_in_results = {r["memory_type"] for r in results}
        assert types_in_results == {"episodic_memory", "foresight"}

        # Verify unified fields
        for r in results:
            assert "id" in r
            assert "score" in r
            assert r["_search_source"] == RetrieveMethod.KEYWORD.value

    @pytest.mark.asyncio
    async def test_unsupported_type_skipped(self, manager):
        """PROFILE is not in ES_REPO_MAP → silently skipped, no error."""
        episodic_hits = [{"_id": "e1", "_score": 1.0}]
        mock_ep_repo = MagicMock()
        mock_ep_repo.multi_search = AsyncMock(return_value=episodic_hits)

        def _bean_router(cls):
            from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
                EpisodicMemoryEsRepository,
            )

            if cls is EpisodicMemoryEsRepository:
                return mock_ep_repo
            raise ValueError(f"Unexpected class: {cls}")

        req = _make_request(
            memory_types=[MemoryType.PROFILE, MemoryType.EPISODIC_MEMORY],
        )

        with (
            patch(f"{_MM}.get_bean_by_type", side_effect=_bean_router),
            patch(f"{_MM}.jieba") as mock_jieba,
            patch(f"{_MM}.filter_stopwords", return_value=["test"]),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            mock_jieba.cut_for_search.return_value = ["test"]
            results = await manager.get_keyword_search_results(req)

        # Only episodic results, PROFILE silently skipped
        assert len(results) == 1
        assert results[0]["memory_type"] == "episodic_memory"

    @pytest.mark.asyncio
    async def test_single_memory_type(self, manager):
        """Single memory type still works exactly like before."""
        hits = [{"_id": "e1", "_score": 2.5}]
        mock_repo = MagicMock()
        mock_repo.multi_search = AsyncMock(return_value=hits)

        req = _make_request(memory_types=[MemoryType.EVENT_LOG])

        with (
            patch(f"{_MM}.get_bean_by_type", return_value=mock_repo),
            patch(f"{_MM}.jieba") as mock_jieba,
            patch(f"{_MM}.filter_stopwords", return_value=["test"]),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            mock_jieba.cut_for_search.return_value = ["test"]
            results = await manager.get_keyword_search_results(req)

        assert len(results) == 1
        assert results[0]["memory_type"] == "event_log"
        mock_repo.multi_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_memory_types(self, manager):
        """Empty memory_types list → no iteration, empty results."""
        req = _make_request(memory_types=[])

        with (
            patch(f"{_MM}.jieba") as mock_jieba,
            patch(f"{_MM}.filter_stopwords", return_value=["test"]),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            mock_jieba.cut_for_search.return_value = ["test"]
            results = await manager.get_keyword_search_results(req)

        assert results == []

    @pytest.mark.asyncio
    async def test_repo_returns_empty(self, manager):
        """Repo returning [] for a type → no items added, no crash."""
        mock_repo = MagicMock()
        mock_repo.multi_search = AsyncMock(return_value=[])

        req = _make_request(memory_types=[MemoryType.EPISODIC_MEMORY])

        with (
            patch(f"{_MM}.get_bean_by_type", return_value=mock_repo),
            patch(f"{_MM}.jieba") as mock_jieba,
            patch(f"{_MM}.filter_stopwords", return_value=[]),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            mock_jieba.cut_for_search.return_value = []
            results = await manager.get_keyword_search_results(req)

        assert results == []


# ---------------------------------------------------------------------------
# 3 & 4. get_vector_search_results
# ---------------------------------------------------------------------------
class TestGetVectorSearchResults:

    @pytest.mark.asyncio
    async def test_multiple_memory_types_merged(self, manager):
        """Multiple memory types → vector results from all repos merged."""
        ep_hits = [{"id": "e1", "score": 0.95}]
        fore_hits = [{"id": "f1", "score": 0.88}]

        mock_ep_repo = MagicMock()
        mock_ep_repo.vector_search = AsyncMock(return_value=ep_hits)
        mock_fore_repo = MagicMock()
        mock_fore_repo.vector_search = AsyncMock(return_value=fore_hits)

        def _bean_router(cls):
            from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
                EpisodicMemoryMilvusRepository,
            )
            from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
                ForesightMilvusRepository,
            )

            if cls is EpisodicMemoryMilvusRepository:
                return mock_ep_repo
            if cls is ForesightMilvusRepository:
                return mock_fore_repo
            raise ValueError(f"Unexpected class: {cls}")

        mock_vectorize = MagicMock()
        mock_vectorize.get_embedding = AsyncMock(
            return_value=_FakeEmbedding([0.1, 0.2, 0.3])
        )

        req = _make_request(
            memory_types=[MemoryType.EPISODIC_MEMORY, MemoryType.FORESIGHT],
        )

        with (
            patch(f"{_MM}.get_bean_by_type", side_effect=_bean_router),
            patch(f"{_MM}.get_vectorize_service", return_value=mock_vectorize),
            patch(f"{_MM}.from_iso_format"),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            results = await manager.get_vector_search_results(req)

        mock_ep_repo.vector_search.assert_awaited_once()
        mock_fore_repo.vector_search.assert_awaited_once()

        assert len(results) == 2
        types_in_results = {r["memory_type"] for r in results}
        assert types_in_results == {"episodic_memory", "foresight"}
        for r in results:
            assert r["_search_source"] == RetrieveMethod.VECTOR.value

    @pytest.mark.asyncio
    async def test_unsupported_type_skipped(self, manager):
        """PROFILE is not in MILVUS_REPO_MAP → silently skipped."""
        ep_hits = [{"id": "e1", "score": 0.9}]
        mock_ep_repo = MagicMock()
        mock_ep_repo.vector_search = AsyncMock(return_value=ep_hits)

        def _bean_router(cls):
            from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
                EpisodicMemoryMilvusRepository,
            )

            if cls is EpisodicMemoryMilvusRepository:
                return mock_ep_repo
            raise ValueError(f"Unexpected class: {cls}")

        mock_vectorize = MagicMock()
        mock_vectorize.get_embedding = AsyncMock(
            return_value=_FakeEmbedding([0.1, 0.2, 0.3])
        )

        req = _make_request(
            memory_types=[MemoryType.PROFILE, MemoryType.EPISODIC_MEMORY],
        )

        with (
            patch(f"{_MM}.get_bean_by_type", side_effect=_bean_router),
            patch(f"{_MM}.get_vectorize_service", return_value=mock_vectorize),
            patch(f"{_MM}.from_iso_format"),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            results = await manager.get_vector_search_results(req)

        assert len(results) == 1
        assert results[0]["memory_type"] == "episodic_memory"

    @pytest.mark.asyncio
    async def test_single_memory_type(self, manager):
        """Single memory type in vector search still works."""
        hits = [{"id": "el1", "score": 0.7}]
        mock_repo = MagicMock()
        mock_repo.vector_search = AsyncMock(return_value=hits)

        mock_vectorize = MagicMock()
        mock_vectorize.get_embedding = AsyncMock(
            return_value=_FakeEmbedding([0.5, 0.6])
        )

        req = _make_request(memory_types=[MemoryType.EVENT_LOG])

        with (
            patch(f"{_MM}.get_bean_by_type", return_value=mock_repo),
            patch(f"{_MM}.get_vectorize_service", return_value=mock_vectorize),
            patch(f"{_MM}.from_iso_format"),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            results = await manager.get_vector_search_results(req)

        assert len(results) == 1
        assert results[0]["memory_type"] == "event_log"
        mock_repo.vector_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_memory_types(self, manager):
        """Empty memory_types list → no Milvus calls, empty results."""
        mock_vectorize = MagicMock()
        mock_vectorize.get_embedding = AsyncMock(
            return_value=_FakeEmbedding([0.1])
        )

        req = _make_request(memory_types=[])

        with (
            patch(f"{_MM}.get_vectorize_service", return_value=mock_vectorize),
            patch(f"{_MM}.from_iso_format"),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            results = await manager.get_vector_search_results(req)

        assert results == []

    @pytest.mark.asyncio
    async def test_embedding_called_once(self, manager):
        """Even with multiple memory types, embedding is computed only once."""
        mock_repo = MagicMock()
        mock_repo.vector_search = AsyncMock(return_value=[])

        mock_vectorize = MagicMock()
        mock_vectorize.get_embedding = AsyncMock(
            return_value=_FakeEmbedding([0.1, 0.2])
        )

        req = _make_request(
            memory_types=[
                MemoryType.EPISODIC_MEMORY,
                MemoryType.FORESIGHT,
                MemoryType.EVENT_LOG,
            ],
        )

        with (
            patch(f"{_MM}.get_bean_by_type", return_value=mock_repo),
            patch(f"{_MM}.get_vectorize_service", return_value=mock_vectorize),
            patch(f"{_MM}.from_iso_format"),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            await manager.get_vector_search_results(req)

        # Embedding should only be computed once regardless of memory type count
        mock_vectorize.get_embedding.assert_awaited_once_with("test query")

    @pytest.mark.asyncio
    async def test_foresight_uses_special_params(self, manager):
        """Foresight vector search passes current_time and time range params."""
        fore_hits = [{"id": "f1", "score": 0.9}]
        mock_fore_repo = MagicMock()
        mock_fore_repo.vector_search = AsyncMock(return_value=fore_hits)

        mock_vectorize = MagicMock()
        mock_vectorize.get_embedding = AsyncMock(
            return_value=_FakeEmbedding([0.1])
        )

        req = RetrieveMemRequest(
            user_id="u1",
            group_id="g1",
            memory_types=[MemoryType.FORESIGHT],
            query="meeting tomorrow",
            top_k=5,
            retrieve_method=RetrieveMethod.VECTOR,
            current_time="2025-06-01T10:00:00",
            start_time="2025-06-01T00:00:00",
            end_time="2025-06-30T23:59:59",
        )

        mock_dt = MagicMock()

        with (
            patch(f"{_MM}.get_bean_by_type", return_value=mock_fore_repo),
            patch(f"{_MM}.get_vectorize_service", return_value=mock_vectorize),
            patch(f"{_MM}.from_iso_format", return_value=mock_dt),
            patch(f"{_MM}.record_retrieve_stage"),
            patch(f"{_MM}.record_retrieve_request"),
            patch(f"{_MM}.record_retrieve_error"),
        ):
            results = await manager.get_vector_search_results(req)

        # Foresight path passes current_time kwarg
        call_kwargs = mock_fore_repo.vector_search.call_args.kwargs
        assert "current_time" in call_kwargs
        assert len(results) == 1
        assert results[0]["memory_type"] == "foresight"
