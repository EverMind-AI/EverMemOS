"""
Foresight Milvus Collection Definition

Foresight-specific Collection class implemented based on MilvusCollectionWithSuffix.
Provides Schema definition and index configuration compatible with ForesightMilvusRepository.
Supports both personal foresight and group foresight.
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import IndexConfig
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection_with_suffix import (
    TenantAwareMilvusCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class ForesightCollection(TenantAwareMilvusCollectionWithSuffix):
    """
    Foresight Milvus Collection

    Supports both personal foresight and group foresight, distinguished by the group_id field.

    Usage:
        # Use Collection
        collection.async_collection().insert([...])
        collection.async_collection().search([...])
    """

    # Base name of the Collection
    _COLLECTION_NAME = "foresight"

    # Collection Schema definition
    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="Unique identifier for foresight",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTORIZE_DIMENSIONS,  # Vector dimension of BAAI/bge-m3 model
                description="Text vector",
            ),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="User ID",
            ),
            FieldSchema(
                name="group_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Group ID",
            ),
            FieldSchema(
                name="participants",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=100,
                max_length=100,
                description="List of related participants",
            ),
            FieldSchema(
                name="parent_type",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Parent memory type (memcell/episode)",
            ),
            FieldSchema(
                name="parent_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Parent memory ID",
            ),
            FieldSchema(
                name="start_time",
                dtype=DataType.INT64,
                description="Foresight start timestamp",
            ),
            FieldSchema(
                name="end_time",
                dtype=DataType.INT64,
                description="Foresight end timestamp",
            ),
            FieldSchema(
                name="duration_days",
                dtype=DataType.INT64,
                description="Duration in days",
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="Foresight content",
            ),
            FieldSchema(
                name="evidence",
                dtype=DataType.VARCHAR,
                max_length=2000,
                description="Evidence supporting this foresight",
            ),
            FieldSchema(
                name="search_content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="Search content (in JSON format)",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=50000,
                description="Detailed information JSON (metadata)",
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
                description="Creation timestamp",
            ),
            FieldSchema(
                name="updated_at", dtype=DataType.INT64, description="Update timestamp"
            ),
        ],
        description="Vector collection for personal foresight",
        enable_dynamic_field=True,
    )

    # Index configuration
    _INDEX_CONFIGS = [
        # Vector field index (for similarity search)
        IndexConfig(
            field_name="vector",
            index_type="HNSW",  # Efficient approximate nearest neighbor search
            metric_type="COSINE",  # Cosine similarity
            params={
                "M": 16,  # Maximum number of connections per node
                "efConstruction": 200,  # Search width during construction
            },
        ),
        # Scalar field indexes (for filtering)
        IndexConfig(
            field_name="user_id",
            index_type="AUTOINDEX",  # Automatically select the most suitable index type
        ),
        IndexConfig(field_name="group_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="parent_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="start_time", index_type="AUTOINDEX"),
        IndexConfig(field_name="end_time", index_type="AUTOINDEX"),
    ]
