"""
AgentSkill Qdrant Repository.

Provides vector search for agent skill records via Qdrant. Supports
cluster-level delete for the replace pattern used by AgentSkillExtractor.

Filter expressions are built as ``qmodels.Filter(must=[FieldCondition...])``
instead of the Milvus string-expression syntax — same semantic, native
typing.
"""

import asyncio
from functools import partial
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qmodels

from core.di.decorators import repository
from core.observation.logger import get_logger
from core.oxm.constants import MAGIC_ALL
from core.oxm.qdrant.base_repository import (
    BaseQdrantRepository,
    compute_effective_threshold,
)
from infra_layer.adapters.out.search.qdrant.memory.agent_skill_collection import (
    AgentSkillCollection,
)

logger = get_logger(__name__)


@repository("agent_skill_qdrant_repository", primary=False)
class AgentSkillQdrantRepository(BaseQdrantRepository[AgentSkillCollection]):
    """
    AgentSkill Qdrant Repository.

    Supports vector similarity search over reusable skill items, plus
    cluster-level deletion for the replace pattern.
    """

    def __init__(self) -> None:
        super().__init__(AgentSkillCollection)

    # ----------------------------------------------------------------- search

    async def vector_search(
        self,
        query_vector: List[float],
        group_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
        maturity_threshold: Optional[float] = 0.6,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search over agent skill items.

        Args:
            query_vector: Query embedding vector.
            group_ids: Group ID list filter (``None`` to skip).
            user_id: User ID filter. ``MAGIC_ALL`` disables the filter.
            cluster_id: Filter by MemScene cluster ID.
            limit: Max results to return.
            score_threshold: Minimum Cosine similarity score (applied
                post-search at the wrapper level; Qdrant also gets it via
                ``score_threshold`` for early stopping).
            radius: Explicit Cosine similarity threshold (>-1.0 enables it).
            maturity_threshold: Minimum maturity score (0.0–1.0). ``None``
                skips the filter (include all maturities).
            confidence_threshold: Minimum confidence score (0.0–1.0). ``None``
                skips the filter.

        Returns:
            List of result dicts with the same shape as the Milvus
            repository for caller parity.
        """
        try:
            conditions: List[qmodels.FieldCondition] = []

            if maturity_threshold is not None:
                conditions.append(
                    qmodels.FieldCondition(
                        key="maturity_score",
                        range=qmodels.Range(gte=maturity_threshold),
                    )
                )

            if confidence_threshold is not None:
                conditions.append(
                    qmodels.FieldCondition(
                        key="confidence",
                        range=qmodels.Range(gte=confidence_threshold),
                    )
                )

            if user_id and user_id != MAGIC_ALL:
                # ``None``/empty user_id means "do not filter" (search across
                # the whole tenant), not "match the empty-string user_id".
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

            if cluster_id:
                conditions.append(
                    qmodels.FieldCondition(
                        key="cluster_id",
                        match=qmodels.MatchValue(value=cluster_id),
                    )
                )

            query_filter = qmodels.Filter(must=conditions) if conditions else None

            ef_value = max(128, limit * 2)
            # Two-stage gating: ``compute_effective_threshold`` returns the
            # more permissive (smaller) positive bound of ``radius`` and
            # ``score_threshold``, or ``None`` if neither is positive. The
            # client-side ``point.score < score_threshold`` post-filter below
            # still enforces the caller's hard cut-off.
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
                        "user_id": payload.get("user_id", ""),
                        "group_id": payload.get("group_id"),
                        "cluster_id": payload.get("cluster_id"),
                        "content": payload.get("content", ""),
                    }
                )

            logger.debug(
                "AgentSkill Qdrant search: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.exception("AgentSkill Qdrant search failed: %s", e)
            raise

    # -------------------------------------------------------- domain deletes

    async def delete_by_cluster_id(self, cluster_id: str) -> int:
        """
        Delete all Qdrant points whose ``cluster_id`` payload matches.

        Used by the AgentSkillExtractor's replace pattern: drop all skills
        of a cluster, then re-upsert the freshly extracted skills.

        Args:
            cluster_id: MemScene cluster ID.

        Returns:
            Number of points deleted (best-effort; Qdrant doesn't return an
            exact count, so we count via a prior scroll).
        """
        try:
            filter_ = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="cluster_id",
                        match=qmodels.MatchValue(value=cluster_id),
                    )
                ]
            )

            client = self.collection.client()
            name = self.collection.name

            # Use Qdrant's ``count`` for an exact total instead of a single
            # scroll page (which could undercount when the cluster has more
            # than the page limit). After counting we issue a single
            # filter-based delete that covers all matches.
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
                logger.debug(
                    "Deleted %d Qdrant points for cluster=%s", count, cluster_id
                )
            return count

        except Exception as e:
            logger.exception(
                "Failed to delete Qdrant points for cluster=%s: %s", cluster_id, e
            )
            # Re-raise so callers can distinguish a genuine zero from an
            # operational failure (consistent with upsert/search/delete_batch
            # in the base repository).
            raise
