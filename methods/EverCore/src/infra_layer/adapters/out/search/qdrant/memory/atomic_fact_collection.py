"""
V1 Atomic Fact Record Qdrant Collection Definition.

Based on MongoDB ``v1_atomic_fact_records``. Stores only search-essential
fields; full data is retrieved from MongoDB using ``parent_id``.

Wire-Layout per point::

    PointStruct(
        id=<str fact_id>,
        vector=<List[float] dim=VECTORIZE_DIMENSIONS>,
        payload={
            "user_id":      str | "",
            "group_id":     str | "",
            "session_id":   str | "",
            "participants": list[str],
            "sender_ids":   list[str],
            "type":         str,         # default RawDataType.CONVERSATION
            "timestamp":    int,         # epoch milliseconds
            "parent_type":  str | "",
            "parent_id":    str | "",
        },
    )
"""

from core.oxm.qdrant.qdrant_collection_base import IndexConfig
from core.tenants.tenantize.oxm.qdrant.tenant_aware_qdrant_collection_with_suffix import (
    TenantAwareQdrantCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class AtomicFactCollection(TenantAwareQdrantCollectionWithSuffix):
    """V1 Atomic Fact Record Qdrant Collection."""

    _COLLECTION_NAME = "v1_atomic_fact_record"

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
            # Type + time-range filters.
            "type": "keyword",
            "timestamp": "integer",
        },
    )
