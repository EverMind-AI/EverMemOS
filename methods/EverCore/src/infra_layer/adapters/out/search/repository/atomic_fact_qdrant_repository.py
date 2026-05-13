"""
Atomic Fact Qdrant Repository.

Provides vector search + batch lookups for atomic-fact records via Qdrant.
Mirrors the Milvus counterpart's surface for caller parity:

- ``create_and_save_atomic_fact``: convenience constructor + upsert
- ``vector_search``: scope + time-range filtered search
- ``batch_vector_search_by_parent_ids``: MRAG-Phase-3 expansion from
  episodes to atomic facts
- ``delete_by_filters``: batch delete by user/group/time-range

Timestamp filter uses **epoch milliseconds** (parity with the Milvus
repository and the AtomicFact converter).
"""

import asyncio
import json
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qmodels

from core.di.decorators import repository
from core.observation.logger import get_logger
from core.oxm.constants import MAGIC_ALL
from core.oxm.qdrant.base_repository import (
    BaseQdrantRepository,
    compute_effective_threshold,
    to_epoch_ms,
)
from infra_layer.adapters.out.search.qdrant.memory.atomic_fact_collection import (
    AtomicFactCollection,
)

logger = get_logger(__name__)


@repository("atomic_fact_qdrant_repository", primary=False)
class AtomicFactQdrantRepository(BaseQdrantRepository[AtomicFactCollection]):
    """V1 Atomic Fact Qdrant Repository."""

    def __init__(self) -> None:
        super().__init__(AtomicFactCollection)

    # ===================================== Document creation / management

    async def create_and_save_atomic_fact(
        self,
        point_id: str,
        user_id: Optional[str],
        atomic_fact: str,
        parent_id: str,
        parent_type: str,
        timestamp: datetime,
        vector: List[float],
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        search_content: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a ``PointStruct`` for an atomic fact and upsert it.

        Returns:
            Summary dict (id / user_id / atomic_fact / parent_* / timestamp /
            search_content) — same shape as the Milvus repository.
        """
        # Explicit None / empty check so a legitimate all-zero embedding
        # is not falsy-rejected (any() on a list of 0.0s is False).
        if vector is None or len(vector) == 0:
            raise ValueError(
                f"Vector is required for AtomicFact {point_id} but was not populated"
            )

        try:
            if search_content is None:
                search_content = [atomic_fact]

            payload = {
                "user_id": user_id or "",
                "group_id": group_id or "",
                "session_id": "",  # not provided by this entry point
                "participants": participants or [],
                "sender_ids": sender_ids or [],
                "type": event_type,
                "timestamp": to_epoch_ms(timestamp),
                "atomic_fact": atomic_fact,
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "parent_type": parent_type,
                "parent_id": parent_id,
            }

            await self.upsert(
                qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
            )

            logger.debug(
                "Atomic fact point upserted: id=%s, user_id=%s", point_id, user_id
            )

            # Result dict keeps the ``id`` key for caller parity with the
            # Milvus counterpart; only the parameter name changed.
            return {
                "id": point_id,
                "user_id": user_id,
                "atomic_fact": atomic_fact,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "timestamp": timestamp,
                "search_content": search_content,
            }

        except Exception as e:
            logger.exception(
                "Failed to create atomic fact point: id=%s, error=%s", point_id, e
            )
            raise

    # ============================================================ search

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search with full scope + time-range filters."""
        try:
            conditions: List[qmodels.FieldCondition] = []

            if user_id and user_id != MAGIC_ALL:
                conditions.append(
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id),
                    )
                )

            if group_ids:
                conditions.append(
                    qmodels.FieldCondition(
                        key="group_id",
                        match=qmodels.MatchAny(any=list(group_ids)),
                    )
                )

            if session_id:
                conditions.append(
                    qmodels.FieldCondition(
                        key="session_id",
                        match=qmodels.MatchValue(value=session_id),
                    )
                )

            if parent_type:
                conditions.append(
                    qmodels.FieldCondition(
                        key="parent_type",
                        match=qmodels.MatchValue(value=parent_type),
                    )
                )

            if parent_id:
                conditions.append(
                    qmodels.FieldCondition(
                        key="parent_id",
                        match=qmodels.MatchValue(value=parent_id),
                    )
                )

            time_range: Dict[str, int] = {}
            if start_time:
                time_range["gte"] = to_epoch_ms(start_time)
            if end_time:
                time_range["lte"] = to_epoch_ms(end_time)
            if time_range:
                conditions.append(
                    qmodels.FieldCondition(
                        key="timestamp",
                        range=qmodels.Range(**time_range),
                    )
                )

            query_filter = qmodels.Filter(must=conditions) if conditions else None
            ef_value = max(128, limit * 2)
            # Two-stage gating — see ``compute_effective_threshold`` for the
            # full precedence rules; ``min(radius, score_threshold)`` is wrong
            # when ``score_threshold`` is at its default ``0.0`` (yields 0,
            # disabling both server- and client-side filtering).
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
                ts_ms = payload.get("timestamp", 0) or 0
                search_results.append(
                    {
                        "id": str(point.id),
                        "score": float(point.score),
                        "user_id": payload.get("user_id"),
                        "group_id": payload.get("group_id"),
                        "session_id": payload.get("session_id"),
                        "participants": payload.get("participants"),
                        # Returned alongside ``search_content`` (and matching
                        # the batch path) so callers don't need a Mongo
                        # round-trip to recover the canonical atomic fact text.
                        "atomic_fact": payload.get("atomic_fact"),
                        # Convert epoch milliseconds back to UTC datetime so
                        # callers get a consistent type across all repository
                        # entry points (parity with create_and_save_atomic_fact
                        # and with the agent_case repository's seconds-path).
                        "timestamp": datetime.fromtimestamp(
                            ts_ms / 1000, tz=timezone.utc
                        ),
                        "parent_type": payload.get("parent_type"),
                        "parent_id": payload.get("parent_id"),
                    }
                )

            logger.debug(
                "AtomicFact Qdrant search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("AtomicFact Qdrant search failed: %s", e)
            raise

    async def batch_vector_search_by_parent_ids(
        self,
        query_vector: List[float],
        parent_ids: List[str],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Vector search restricted to a list of ``parent_id`` values.

        Used by MRAG Phase 3 to expand episodes into their atomic facts.
        Total effective limit is ``limit * len(parent_ids)``.
        """
        if not parent_ids:
            return []

        try:
            conditions: List[qmodels.FieldCondition] = [
                qmodels.FieldCondition(
                    key="parent_id",
                    match=qmodels.MatchAny(any=list(parent_ids)),
                )
            ]

            if user_id and user_id != MAGIC_ALL:
                conditions.append(
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id),
                    )
                )

            if group_ids:
                conditions.append(
                    qmodels.FieldCondition(
                        key="group_id",
                        match=qmodels.MatchAny(any=list(group_ids)),
                    )
                )

            total_limit = limit * len(parent_ids)
            ef_value = max(128, total_limit * 2)

            scored_points = await self.search(
                query_vector=query_vector,
                limit=total_limit,
                query_filter=qmodels.Filter(must=conditions),
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold if score_threshold > 0 else None,
                search_params=qmodels.SearchParams(hnsw_ef=ef_value),
            )

            search_results: List[Dict[str, Any]] = []
            for point in scored_points:
                if point.score < score_threshold:
                    continue
                payload = point.payload or {}
                ts_ms = payload.get("timestamp", 0) or 0
                search_results.append(
                    {
                        "id": str(point.id),
                        "score": float(point.score),
                        "user_id": payload.get("user_id"),
                        "group_id": payload.get("group_id"),
                        "parent_type": payload.get("parent_type"),
                        "parent_id": payload.get("parent_id"),
                        "atomic_fact": payload.get("atomic_fact"),
                        "timestamp": datetime.fromtimestamp(
                            ts_ms / 1000, tz=timezone.utc
                        ),
                        "participants": payload.get("participants"),
                    }
                )

            logger.debug(
                "AtomicFact batch search by parent_ids: parent_ids=%d, results=%d",
                len(parent_ids),
                len(search_results),
            )
            return search_results

        except Exception as e:
            logger.error("AtomicFact batch search by parent_ids failed: %s", e)
            raise

    # ========================================================== deletion

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Batch delete by filter combination; at least one filter required."""
        try:
            conditions: List[qmodels.FieldCondition] = []

            if user_id != MAGIC_ALL:
                conditions.append(
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id or ""),
                    )
                )
            if group_id != MAGIC_ALL:
                conditions.append(
                    qmodels.FieldCondition(
                        key="group_id",
                        match=qmodels.MatchValue(value=group_id or ""),
                    )
                )

            time_range: Dict[str, int] = {}
            if start_time:
                time_range["gte"] = to_epoch_ms(start_time)
            if end_time:
                time_range["lte"] = to_epoch_ms(end_time)
            if time_range:
                conditions.append(
                    qmodels.FieldCondition(
                        key="timestamp",
                        range=qmodels.Range(**time_range),
                    )
                )

            if not conditions:
                raise ValueError("At least one filter condition must be provided")

            filter_ = qmodels.Filter(must=conditions)
            client = self.collection.client()
            name = self.collection.name

            # Exact count for the deleted-points return value (avoids
            # the bounded scroll-page undercount).
            count_result = await asyncio.to_thread(
                partial(
                    client.count,
                    collection_name=name,
                    count_filter=filter_,
                    exact=True,
                )
            )
            delete_count = count_result.count

            if delete_count > 0:
                await asyncio.to_thread(
                    partial(
                        client.delete,
                        collection_name=name,
                        points_selector=qmodels.FilterSelector(filter=filter_),
                        wait=True,
                    )
                )

            logger.debug(
                "Batch deleted atomic facts: deleted %d points", delete_count
            )
            return delete_count

        except Exception as e:
            logger.error("Failed to batch delete atomic facts: %s", e)
            raise
