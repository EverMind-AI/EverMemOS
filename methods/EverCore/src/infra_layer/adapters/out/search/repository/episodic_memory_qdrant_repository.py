"""
Episodic Memory Qdrant Repository.

V1 simplified repository for vector semantic retrieval. Only stores
search-essential fields in Qdrant; full data is fetched from MongoDB via
``parent_id`` back-reference.

Mirrors the surface of the Milvus counterpart for caller parity, but uses
native Qdrant filtering (``qmodels.Filter(must=[FieldCondition...])``)
instead of Milvus' string expression syntax.
"""

import asyncio
import json
from datetime import datetime
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
from infra_layer.adapters.out.search.qdrant.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)

logger = get_logger(__name__)


@repository("episodic_memory_qdrant_repository", primary=False)
class EpisodicMemoryQdrantRepository(BaseQdrantRepository[EpisodicMemoryCollection]):
    """V1 simplified Qdrant repository for episodic memory."""

    def __init__(self) -> None:
        super().__init__(EpisodicMemoryCollection)

    # ===================================== Document creation / management

    async def create_and_save_episodic_memory(
        self,
        id: str,
        user_id: str,
        timestamp: datetime,
        episode: str,
        search_content: List[str],
        vector: List[float],
        title: Optional[str] = None,
        summary: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        subject: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level convenience constructor: build a ``PointStruct`` and upsert.

        Returns:
            A small summary dict (id, user_id, timestamp, episode,
            search_content) — same shape as the Milvus repository to keep
            callers untouched at cutover.

        Raises:
            ValueError: when ``vector`` is None or empty. A missing embedding
                would only surface as a confusing 400 from Qdrant at upsert
                time, far from the bad caller. Fail fast instead.
        """
        if vector is None or len(vector) == 0:
            raise ValueError(
                f"Vector is required for EpisodicMemory {id} but was not populated"
            )

        try:
            payload = {
                "user_id": user_id or "",
                "group_id": group_id or "",
                "session_id": "",  # not provided by this entry point
                "participants": participants or [],
                "sender_ids": sender_ids or [],
                "type": event_type or "",
                "timestamp": to_epoch_ms(timestamp),
                "episode": episode,
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "parent_type": parent_type or "",
                "parent_id": parent_id or "",
            }

            await self.upsert(
                qmodels.PointStruct(id=id, vector=vector, payload=payload)
            )

            logger.debug(
                "Episodic memory point upserted: id=%s, user_id=%s", id, user_id
            )

            return {
                "id": id,
                "user_id": user_id,
                "timestamp": timestamp,
                "episode": episode,
                "search_content": search_content,
            }

        except Exception as e:
            logger.error(
                "Failed to create episodic memory point: id=%s, error=%s", id, e
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
        """Vector similarity search with optional scope + time-range filters."""
        try:
            conditions: List[qmodels.FieldCondition] = []

            # Guard both ``None`` (no scope passed) and the explicit "all"
            # sentinel. Without the ``is not None`` guard a default ``user_id=None``
            # slipped past the sentinel check and the ``user_id or ""`` fallback
            # filtered the search to documents with an empty ``user_id`` payload,
            # i.e. zero hits in practice.
            if user_id is not None and user_id != MAGIC_ALL:
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
            # Two-stage score gating (parity with Milvus repository):
            #   - Server-side: pass the *more permissive* (lower) of
            #     ``radius`` and ``score_threshold`` so Qdrant returns the
            #     wider net.
            #   - Client-side: the ``point.score < score_threshold`` post-
            #     filter enforces the hard caller-facing minimum.
            # This way callers can use ``radius`` to widen recall without
            # accidentally making the server-side cut stricter than the
            # caller's own cut-off.
            # Two-stage gating — see ``compute_effective_threshold`` for the
            # precedence rules. The plain ``min(radius, score_threshold)``
            # collapsed to ``0`` whenever ``score_threshold`` was at its
            # default and silently disabled both server- and client-side
            # filtering.
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
                        # Normalise epoch ms back to a UTC ``datetime`` for
                        # caller parity with the other Qdrant repositories
                        # (atomic_fact, agent_case, foresight) — they all
                        # surface time as ``datetime``, returning the raw
                        # epoch here used to break callers that wanted a
                        # single time type across collections.
                        "timestamp": datetime.fromtimestamp(
                            ts_ms / 1000, tz=timezone.utc
                        ),
                        "parent_type": payload.get("parent_type"),
                        "parent_id": payload.get("parent_id"),
                        "type": payload.get("type"),
                        "episode": payload.get("episode"),
                    }
                )

            logger.debug(
                "EpisodicMemory Qdrant search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.exception("EpisodicMemory Qdrant search failed: %s", e)
            raise

    # ========================================================== deletion

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Batch delete by filter combination.

        At least one filter (other than ``MAGIC_ALL`` sentinels) must be
        provided, matching the Milvus repository's guard.
        """
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

            # Use Qdrant's exact ``count`` instead of a bounded scroll page,
            # so the returned count reflects the *full* number of points
            # the filter matches (a 10k scroll cap would undercount large
            # tenants and produce a misleading return value).
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
                "Batch deleted episodic memories: deleted %d points", delete_count
            )
            return delete_count

        except Exception as e:
            logger.error("Failed to batch delete episodic memories: %s", e)
            raise
