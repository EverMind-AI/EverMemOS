"""
V1 Foresight Record Qdrant Collection Definition.

Based on MongoDB ``v1_foresight_records``. Stores only search-essential
fields; full data is retrieved from MongoDB using ``parent_id``.

Wire-Layout per point::

    PointStruct(
        id=<str foresight_id>,
        vector=<List[float] dim=VECTORIZE_DIMENSIONS>,
        payload={
            "user_id":        str | "",
            "group_id":       str | "",
            "session_id":     str | "",
            "participants":   list[str],
            "sender_ids":     list[str],
            "type":           str | "",
            "start_time":     int,           # epoch milliseconds
            "end_time":       int,           # epoch milliseconds
            "duration_days":  int,
            "content":        str | None,
            "evidence":       str | "",
            "search_content": str,           # JSON-stringified list
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


class ForesightCollection(TenantAwareQdrantCollectionWithSuffix):
    """V1 Foresight Record Qdrant Collection."""

    _COLLECTION_NAME = "v1_foresight_record"

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
            "start_time": "integer",
            "end_time": "integer",
        },
    )
