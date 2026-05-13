"""
Foresight Qdrant Repository.

V1 repository for vector semantic retrieval over foresight records.
Mirrors the Milvus counterpart's surface (``create_and_save_foresight_mem``,
``vector_search``, ``delete_by_filters``) for caller parity.

**Note on time filters:** the Foresight schema stores ``start_time`` and
``end_time`` (both epoch milliseconds). Both ``vector_search`` and
``delete_by_filters`` use **window-overlap** semantics — a record matches
when its window ``[start_time, end_time]`` overlaps the query window
``[start_time arg, end_time arg]``:

- ``end_time`` arg -> ``payload.start_time <= end_time_ms``
- ``start_time`` arg -> ``payload.end_time >= start_time_ms``

(Older revisions of this file used the inverted containment predicates,
which silently dropped partially-overlapping records on read and left
them undeleted on cleanup.)
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
from infra_layer.adapters.out.search.qdrant.memory.foresight_collection import (
    ForesightCollection,
)

logger = get_logger(__name__)


@repository("foresight_qdrant_repository", primary=False)
class ForesightQdrantRepository(BaseQdrantRepository[ForesightCollection]):
    """V1 Foresight Qdrant Repository."""

    def __init__(self) -> None:
        super().__init__(ForesightCollection)

    # ===================================== Document creation / management

    async def create_and_save_foresight_mem(
        self,
        id: str,
        user_id: Optional[str],
        content: str,
        parent_id: str,
        parent_type: str,
        vector: List[float],
        group_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_days: Optional[int] = None,
        evidence: Optional[str] = None,
        search_content: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a ``PointStruct`` for a foresight record and upsert it.

        ``session_id`` is written into the payload so the matching
        ``vector_search(session_id=...)`` filter can hit. Parity with the
        Foresight schema's ``session_id`` payload index.

        Returns a caller-compatible summary dict (same shape as the Milvus
        repository for cutover).
        """
        if not vector:
            raise ValueError(
                f"Vector is required for Foresight {id} but was not populated"
            )

        try:
            if search_content is None:
                search_content = [content]
                if evidence:
                    search_content.append(evidence)

            payload = {
                "user_id": user_id or "",
                "group_id": group_id or "",
                "session_id": session_id or "",
                "participants": participants or [],
                "sender_ids": sender_ids or [],
                "type": event_type,
                # ``None`` (not 0) for missing bounds so range queries treat
                # "no start/end" distinct from "epoch 1970". Downstream
                # filters skip the field when payload value is None.
                "start_time": to_epoch_ms(start_time) if start_time else None,
                "end_time": to_epoch_ms(end_time) if end_time else None,
                "duration_days": duration_days or 0,
                "content": content,
                "evidence": evidence or "",
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "parent_type": parent_type,
                "parent_id": parent_id,
            }

            await self.upsert(
                qmodels.PointStruct(id=id, vector=vector, payload=payload)
            )

            logger.debug(
                "Foresight point upserted: id=%s, user_id=%s", id, user_id
            )

            return {
                "id": id,
                "user_id": user_id,
                "content": content,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "search_content": search_content,
            }

        except Exception as e:
            logger.error(
                "Failed to create foresight point: id=%s, error=%s", id, e
            )
            raise

    # ============================================================ search

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        sender_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search with scope, sender, and time-range filters.

        Time filters semantic — **window-overlap**, not containment. A
        record matches when its window ``[start_time, end_time]`` overlaps
        the query window ``[start_time arg, end_time arg]``:

        - ``end_time`` arg -> ``payload.start_time <= end_time_ms`` (record
          starts on or before the query window ends).
        - ``start_time`` arg -> ``payload.end_time >= start_time_ms`` (record
          ends on or after the query window starts).

        The previous containment filter (``start>=q.start AND end<=q.end``)
        silently dropped foresights whose window only partially overlapped
        the query window, which is rarely what callers want.

        ``sender_id`` filters via Qdrant's array-containment semantics on the
        ``sender_ids`` payload field.
        """
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

            if sender_id:
                # Qdrant matches arrays element-wise on MatchValue, so this
                # is the equivalent of Milvus' ``array_contains(sender_ids, x)``.
                conditions.append(
                    qmodels.FieldCondition(
                        key="sender_ids",
                        match=qmodels.MatchValue(value=sender_id),
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

            # Window-overlap filter (see docstring): a record overlaps the
            # query window iff record.end_time >= query.start AND
            # record.start_time <= query.end.
            if start_time:
                conditions.append(
                    qmodels.FieldCondition(
                        key="end_time",
                        range=qmodels.Range(gte=to_epoch_ms(start_time)),
                    )
                )
            if end_time:
                conditions.append(
                    qmodels.FieldCondition(
                        key="start_time",
                        range=qmodels.Range(lte=to_epoch_ms(end_time)),
                    )
                )

            query_filter = qmodels.Filter(must=conditions) if conditions else None
            ef_value = max(128, limit * 2)
            # Two-stage gating — see ``compute_effective_threshold``.
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
                search_results.append(
                    {
                        "id": str(point.id),
                        "score": float(point.score),
                        "user_id": payload.get("user_id"),
                        "group_id": payload.get("group_id"),
                        "sender_ids": payload.get("sender_ids"),
                        "session_id": payload.get("session_id"),
                        "participants": payload.get("participants"),
                        "start_time": payload.get("start_time"),
                        "end_time": payload.get("end_time"),
                        "parent_type": payload.get("parent_type"),
                        "parent_id": payload.get("parent_id"),
                    }
                )

            logger.debug(
                "Foresight Qdrant search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("Foresight Qdrant search failed: %s", e)
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
        Batch delete by filter combination; at least one filter required.

        Time-range semantic matches ``vector_search``: ``start_time`` arg
        maps to ``payload.start_time >= ...``, ``end_time`` arg maps to
        ``payload.end_time <= ...``.
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

            # Use the same window-overlap semantics as ``vector_search``
            # (record overlaps query window when record.end >= q.start AND
            # record.start <= q.end). Diverging here would silently keep
            # foresights that ``vector_search`` already returns, leaving
            # callers with stale records after a "delete this window" call.
            if start_time:
                conditions.append(
                    qmodels.FieldCondition(
                        key="end_time",
                        range=qmodels.Range(gte=to_epoch_ms(start_time)),
                    )
                )
            if end_time:
                conditions.append(
                    qmodels.FieldCondition(
                        key="start_time",
                        range=qmodels.Range(lte=to_epoch_ms(end_time)),
                    )
                )

            if not conditions:
                raise ValueError("At least one filter condition must be provided")

            filter_ = qmodels.Filter(must=conditions)
            client = self.collection.client()
            name = self.collection.name

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
                "Batch deleted foresights: deleted %d points", delete_count
            )
            return delete_count

        except Exception as e:
            logger.error("Failed to batch delete foresights: %s", e)
            raise
