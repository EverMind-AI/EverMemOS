"""
V1 Episodic Memory Qdrant Collection Definition.

Based on MongoDB ``v1_episodic_memories``. Stores only search-essential
fields; full data is retrieved from MongoDB using ``parent_id``. Qdrant is
schema-flexible — payload fields are described here for documentation
only; only fields that need filtering get an explicit payload index.

Wire-Layout per point::

    PointStruct(
        id=<str event_id>,
        vector=<List[float] dim=VECTORIZE_DIMENSIONS>,
        payload={
            "user_id":        str | "",
            "group_id":       str | "",
            "session_id":     str | "",
            "participants":   list[str],
            "sender_ids":     list[str],
            "type":           str | "",
            "timestamp":      int,        # epoch milliseconds
            "episode":        str | "",
            "search_content": str,        # JSON-stringified list
            "parent_type":    str | "",
            "parent_id":      str | "",
        },
    )
"""

from core.oxm.qdrant.qdrant_collection_base import IndexConfig
from core.tenants.tenantize.oxm.qdrant.tenant_aware_qdrant_collection_with_suffix import (
    TenantAwareQdrantCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class EpisodicMemoryCollection(TenantAwareQdrantCollectionWithSuffix):
    """
    V1 Episodic Memory Qdrant Collection.

    Tenant-prefixed name resolution comes from
    ``TenantAwareQdrantCollectionWithSuffix`` (e.g.,
    ``acme_v1_episodic_memory``). HNSW parameters are tuned conservatively
    for ~10k-scale collections; revisit for larger workloads.
    """

    # Logical base name. The actual Qdrant collection name is resolved at
    # construction time by the parent class (tenant prefix + optional suffix).
    _COLLECTION_NAME = "v1_episodic_memory"

    _VECTOR_PARAMS = IndexConfig(
        size=VECTORIZE_DIMENSIONS,
        distance="cosine",
        hnsw_m=16,
        hnsw_ef_construct=200,
        payload_indexes={
            # Tenant-isolation + scope filters (all keyword for exact-equality).
            "user_id": "keyword",
            "group_id": "keyword",
            "session_id": "keyword",
            # Back-reference filters (lookup-by-parent for resync flows).
            "parent_id": "keyword",
            "parent_type": "keyword",
            # Type and time-range filters used by the search service.
            "type": "keyword",
            "timestamp": "integer",
        },
    )
