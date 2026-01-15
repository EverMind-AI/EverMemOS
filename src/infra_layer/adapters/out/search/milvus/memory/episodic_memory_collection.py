"""
Episodic Memory Milvus Collection Definition

An episodic memory dedicated Collection class implemented based on MilvusCollectionWithSuffix.
Provides Schema definition and index configuration compatible with EpisodicMemoryMilvusRepository.
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import IndexConfig
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection_with_suffix import (
    TenantAwareMilvusCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class EpisodicMemoryCollection(TenantAwareMilvusCollectionWithSuffix):
    """
    Episodic Memory Milvus Collection

    Usage:
        # Use the Collection
        collection.async_collection().insert([...])
        collection.async_collection().search([...])
    """

    # Base name for the Collection
    _COLLECTION_NAME = "episodic_memory"

    # Collection Schema definition
    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="Event unique identifier",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTORIZE_DIMENSIONS,  # Vector dimension of the BAAI/bge-m3 model
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
                description="List of participants (used for user filtering in group memory)",
            ),
            FieldSchema(
                name="event_type",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="Event type (e.g., conversation, email, etc.)",
            ),
            FieldSchema(
                name="timestamp", dtype=DataType.INT64, description="Event timestamp"
            ),
            FieldSchema(
                name="episode",
                dtype=DataType.VARCHAR,
                max_length=10000,
                description="Episode description",
            ),
            FieldSchema(
                name="search_content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="Search content",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=50000,
                description="Detailed non-retrieval information in JSON (metadata)",
            ),
            FieldSchema(
                name="parent_type",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Parent memory type (e.g., memcell)",
            ),
            FieldSchema(
                name="parent_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Parent memory ID",
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
        description="Vector collection for episodic memory",
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
        IndexConfig(field_name="event_type", index_type="AUTOINDEX"),
        IndexConfig(field_name="parent_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="timestamp", index_type="AUTOINDEX"),
    ]
