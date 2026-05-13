"""
AgentCase Qdrant Collection Definition.

Stores vector embeddings of agent task-solving experiences. The vector
represents the ``task_intent`` of one experience per MemCell. Full payload
is fetched from MongoDB via the ``parent_id`` back-reference.

Wire-Layout per point::

    PointStruct(
        id=<str AgentCaseRecord id>,
        vector=<List[float] dim=VECTORIZE_DIMENSIONS>,
        payload={
            "user_id":     str | "",
            "group_id":    str | "",
            "session_id":  str | "",
            "timestamp":   int,      # epoch seconds
            "task_intent": str,      # truncated to 5000 chars
            "parent_type": str | "",
            "parent_id":   str | "",
        },
    )
"""

from core.oxm.qdrant.qdrant_collection_base import IndexConfig
from core.tenants.tenantize.oxm.qdrant.tenant_aware_qdrant_collection_with_suffix import (
    TenantAwareQdrantCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class AgentCaseCollection(TenantAwareQdrantCollectionWithSuffix):
    """V1 Agent Case Qdrant Collection. Tenant-prefixed at construction time."""

    _COLLECTION_NAME = "v1_agent_case"

    _VECTOR_PARAMS = IndexConfig(
        size=VECTORIZE_DIMENSIONS,
        distance="cosine",
        hnsw_m=16,
        hnsw_ef_construct=200,
        payload_indexes={
            # Scope filters.
            "user_id": "keyword",
            "group_id": "keyword",
            "session_id": "keyword",
            # Back-reference filters.
            "parent_id": "keyword",
            "parent_type": "keyword",
            # Time-range filter.
            "timestamp": "integer",
        },
    )
