"""
V1 User Profile Qdrant Collection Definition.

Based on MongoDB ``v1_user_profiles``. Stores per-item embeddings — one
point per ``explicit_info`` entry and one per ``implicit_trait`` (the
converter splits a single Mongo doc into many points). UserProfile does
**not** have ``session_id`` (user-level aggregation).

Wire-Layout per point::

    PointStruct(
        id=<str, freshly minted ObjectId per item>,
        vector=<List[float] dim=VECTORIZE_DIMENSIONS>,
        payload={
            "user_id":       str,          # required
            "group_id":      str | "",
            "scenario":      str,          # "solo" | "team"
            "memcell_count": int,
            "item_type":     str,          # "explicit_info" | "implicit_trait"
            "embed_text":    str,          # text used to generate the vector
        },
    )
"""

from core.oxm.qdrant.qdrant_collection_base import IndexConfig
from core.tenants.tenantize.oxm.qdrant.tenant_aware_qdrant_collection_with_suffix import (
    TenantAwareQdrantCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class UserProfileCollection(TenantAwareQdrantCollectionWithSuffix):
    """V1 User Profile Qdrant Collection."""

    _COLLECTION_NAME = "v1_user_profile"

    _VECTOR_PARAMS = IndexConfig(
        size=VECTORIZE_DIMENSIONS,
        distance="cosine",
        hnsw_m=16,
        hnsw_ef_construct=200,
        payload_indexes={
            # Scope filters.
            "user_id": "keyword",
            "group_id": "keyword",
            # Cohort filters.
            "scenario": "keyword",
            "item_type": "keyword",
        },
    )
