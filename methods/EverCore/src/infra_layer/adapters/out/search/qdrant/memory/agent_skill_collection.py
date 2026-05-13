"""
AgentSkill Qdrant Collection Definition.

Stores vector embeddings of reusable skill items. The vector represents the
embedding of name + description; ``content`` is the primary searchable text.

Wire-Layout per point::

    PointStruct(
        id=<str AgentSkillRecord id>,
        vector=<List[float] dim=VECTORIZE_DIMENSIONS>,
        payload={
            "user_id":        str | "",     # agent owner
            "group_id":       str | "",
            "cluster_id":     str | "",     # MemScene cluster id
            "content":        str,          # name + "\\n" + description, ≤5000 chars
            "maturity_score": float,        # 0.0–1.0
            "confidence":     float,        # 0.0–1.0
        },
    )
"""

from core.oxm.qdrant.qdrant_collection_base import IndexConfig
from core.tenants.tenantize.oxm.qdrant.tenant_aware_qdrant_collection_with_suffix import (
    TenantAwareQdrantCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class AgentSkillCollection(TenantAwareQdrantCollectionWithSuffix):
    """V1 Agent Skill Qdrant Collection."""

    _COLLECTION_NAME = "v1_agent_skill"

    _VECTOR_PARAMS = IndexConfig(
        size=VECTORIZE_DIMENSIONS,
        distance="cosine",
        hnsw_m=16,
        hnsw_ef_construct=200,
        payload_indexes={
            # Scope filters.
            "user_id": "keyword",
            "group_id": "keyword",
            "cluster_id": "keyword",
            # Quality-score filters (range queries for thresholding).
            "maturity_score": "float",
            "confidence": "float",
        },
    )
