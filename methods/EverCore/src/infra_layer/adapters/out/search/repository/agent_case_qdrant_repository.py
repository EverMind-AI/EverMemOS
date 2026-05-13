"""
AgentCase Qdrant Repository.

Provides vector search for agent task-solving experiences via Qdrant.
Mirrors the Milvus counterpart's surface for caller parity.

Timestamp filter is in **epoch seconds** (parity with the Milvus repository
and the AgentCase converter — both store seconds, not milliseconds, for this
collection).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qmodels

from core.di.decorators import repository
from core.observation.logger import get_logger
from core.oxm.constants import MAGIC_ALL
from core.oxm.qdrant.base_repository import (
    BaseQdrantRepository,
    compute_effective_threshold,
    to_epoch_s,
)
from infra_layer.adapters.out.search.qdrant.memory.agent_case_collection import (
    AgentCaseCollection,
)

logger = get_logger(__name__)


@repository("agent_case_qdrant_repository", primary=False)
class AgentCaseQdrantRepository(BaseQdrantRepository[AgentCaseCollection]):
    """V1 AgentCase Qdrant Repository."""

    def __init__(self) -> None:
        super().__init__(AgentCaseCollection)

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        parent_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search with scope + time-range filters."""
        try:
            conditions: List[qmodels.FieldCondition] = []

            if user_id and user_id != MAGIC_ALL:
                # ``None``/empty user_id means "do not filter".
                conditions.append(
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id),
                    )
                )

            if session_id:
                conditions.append(
                    qmodels.FieldCondition(
                        key="session_id",
                        match=qmodels.MatchValue(value=session_id),
                    )
                )

            if group_ids:
                conditions.append(
                    qmodels.FieldCondition(
                        key="group_id",
                        match=qmodels.MatchAny(any=list(group_ids)),
                    )
                )

            if parent_id:
                conditions.append(
                    qmodels.FieldCondition(
                        key="parent_id",
                        match=qmodels.MatchValue(value=parent_id),
                    )
                )

            # AgentCase timestamps are epoch SECONDS (Milvus parity).
            # to_epoch_s coerces tz-naive datetimes to UTC to avoid silent
            # locale drift in the filter bounds.
            time_range: Dict[str, int] = {}
            if start_time:
                time_range["gte"] = to_epoch_s(start_time)
            if end_time:
                time_range["lte"] = to_epoch_s(end_time)
            if time_range:
                conditions.append(
                    qmodels.FieldCondition(
                        key="timestamp",
                        range=qmodels.Range(**time_range),
                    )
                )

            query_filter = qmodels.Filter(must=conditions) if conditions else None
            ef_value = max(128, limit * 2)
            # Two-stage gating: use the more permissive (lower) positive
            # bound of ``radius`` and ``score_threshold`` server-side, then
            # enforce the hard caller cut client-side. See
            # ``compute_effective_threshold`` for the precedence rules.
            effective_threshold = compute_effective_threshold(
                radius, score_threshold
            )

            scored_points = await self.search(
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                score_threshold=effective_threshold,
                search_params=qmodels.SearchParams(hnsw_ef=ef_value),
            )

            search_results: List[Dict[str, Any]] = []
            for point in scored_points:
                if point.score < score_threshold:
                    continue
                payload = point.payload or {}
                ts_seconds = payload.get("timestamp", 0) or 0
                search_results.append(
                    {
                        "id": str(point.id),
                        "score": float(point.score),
                        "user_id": payload.get("user_id"),
                        "group_id": payload.get("group_id"),
                        "session_id": payload.get("session_id", ""),
                        # Convert epoch seconds back to UTC datetime for caller
                        # parity with the Milvus repository.
                        "timestamp": datetime.fromtimestamp(
                            ts_seconds, tz=timezone.utc
                        ),
                        "task_intent": payload.get("task_intent", ""),
                        "parent_type": payload.get("parent_type", ""),
                        "parent_id": payload.get("parent_id", ""),
                    }
                )

            logger.debug(
                "AgentCase Qdrant search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.exception("AgentCase Qdrant search failed: %s", e)
            raise
