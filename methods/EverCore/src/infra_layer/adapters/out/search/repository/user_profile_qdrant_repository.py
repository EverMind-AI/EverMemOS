"""
User Profile Qdrant Repository.

V1 repository for vector semantic retrieval over user-profile items.
Mirrors the Milvus counterpart's surface for caller parity:
- ``vector_search``: scope (user_id/group_id) + scenario filter
- ``delete_by_user_group``: drop all profile items for a (user_id, group_id) pair

Note: UserProfile has no ``session_id`` (user-level aggregation).
"""

import asyncio
from functools import partial
from hashlib import sha256
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qmodels

from core.di.decorators import repository
from core.observation.logger import get_logger
from core.oxm.constants import MAGIC_ALL
from core.oxm.qdrant.base_repository import (
    BaseQdrantRepository,
    compute_effective_threshold,
)
from infra_layer.adapters.out.search.qdrant.memory.user_profile_collection import (
    UserProfileCollection,
)

logger = get_logger(__name__)


def _fp(value: Optional[str]) -> str:
    """
    Short fingerprint for log lines. Profile identifiers can be PII when
    the upstream caller is using human-readable user/group ids; emitting
    the raw value into centralised logs is an unnecessary compliance risk.
    A 12-char SHA-256 prefix is enough to correlate events without
    surfacing the underlying identifier. ``None``/empty becomes ``"-"``.
    """
    if not value:
        return "-"
    return sha256(value.encode("utf-8")).hexdigest()[:12]


@repository("user_profile_qdrant_repository", primary=False)
class UserProfileQdrantRepository(BaseQdrantRepository[UserProfileCollection]):
    """V1 User Profile Qdrant Repository."""

    def __init__(self) -> None:
        super().__init__(UserProfileCollection)

    # ============================================================ search

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        scenario: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search with scope + scenario filters."""
        try:
            conditions: List[qmodels.FieldCondition] = []

            if user_id and user_id != MAGIC_ALL:
                conditions.append(
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id),
                    )
                )

            if group_id and group_id != MAGIC_ALL:
                conditions.append(
                    qmodels.FieldCondition(
                        key="group_id",
                        match=qmodels.MatchValue(value=group_id),
                    )
                )

            if scenario and scenario != MAGIC_ALL:
                conditions.append(
                    qmodels.FieldCondition(
                        key="scenario",
                        match=qmodels.MatchValue(value=scenario),
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
                        "scenario": payload.get("scenario"),
                        "memcell_count": payload.get("memcell_count"),
                        "item_type": payload.get("item_type", ""),
                        "embed_text": payload.get("embed_text", ""),
                    }
                )

            logger.debug(
                "UserProfile Qdrant search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("UserProfile Qdrant search failed: %s", e)
            raise

    # ========================================================== deletion

    async def delete_by_user_group(self, user_id: str, group_id: str) -> int:
        """
        Delete all profile items for a (user_id, group_id) pair.

        Returns the actual point count via ``client.count(exact=True)``
        (Qdrant's filter-based delete doesn't return a count of its own).
        """
        try:
            filter_ = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id),
                    ),
                    qmodels.FieldCondition(
                        key="group_id",
                        match=qmodels.MatchValue(value=group_id),
                    ),
                ]
            )
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
            count = count_result.count

            if count > 0:
                await asyncio.to_thread(
                    partial(
                        client.delete,
                        collection_name=name,
                        points_selector=qmodels.FilterSelector(filter=filter_),
                        wait=True,
                    )
                )

            logger.info(
                "Deleted profile items: user_fp=%s group_fp=%s count=%d",
                _fp(user_id),
                _fp(group_id),
                count,
            )
            return count

        except Exception as e:
            logger.exception(
                "Failed to delete profile items: user_fp=%s group_fp=%s error=%s",
                _fp(user_id),
                _fp(group_id),
                e,
            )
            # Re-raise so callers can distinguish "nothing to delete" from
            # an operational failure (consistent with base_repository fix).
            raise
