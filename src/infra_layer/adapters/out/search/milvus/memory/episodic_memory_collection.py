"""
情景记忆 Milvus Collection 定义

基于 MilvusCollectionWithSuffix 实现的情景记忆专用 Collection 类。
提供了与 EpisodicMemoryMilvusRepository 兼容的 Schema 定义和索引配置。
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import IndexConfig
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection_with_suffix import (
    TenantAwareMilvusCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class EpisodicMemoryCollection(TenantAwareMilvusCollectionWithSuffix):
    """
    情景记忆 Milvus Collection

    使用方式：
        # 使用 Collection
        collection.async_collection().insert([...])
        collection.async_collection().search([...])
    """

    # Collection 基础名称
    _COLLECTION_NAME = "episodic_memory"

    # Collection Schema 定义
    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="事件唯一标识",
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=VECTORIZE_DIMENSIONS,  # BAAI/bge-m3 模型的向量维度
                description="文本向量",
            ),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="用户ID",
            ),
            FieldSchema(
                name="group_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="群组ID",
            ),
            FieldSchema(
                name="participants",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=100,
                max_length=100,
                description="参与者列表（用于群组记忆的用户过滤）",
            ),
            FieldSchema(
                name="event_type",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="事件类型（如 conversation, email 等）",
            ),
            FieldSchema(
                name="timestamp", dtype=DataType.INT64, description="事件时间戳"
            ),
            FieldSchema(
                name="episode",
                dtype=DataType.VARCHAR,
                max_length=10000,
                description="情景描述",
            ),
            FieldSchema(
                name="search_content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="搜索内容",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=50000,
                description="非检索用的详细信息JSON（元数据）",
            ),
            FieldSchema(
                name="created_at", dtype=DataType.INT64, description="创建时间戳"
            ),
            FieldSchema(
                name="updated_at", dtype=DataType.INT64, description="更新时间戳"
            ),
        ],
        description="情景记忆向量集合",
        enable_dynamic_field=True,
    )

    # 索引配置
    _INDEX_CONFIGS = [
        # 向量字段索引（用于相似度搜索）
        IndexConfig(
            field_name="vector",
            index_type="HNSW",  # 高效的近似最近邻搜索
            metric_type="COSINE",  # 余弦相似度
            params={
                "M": 16,  # 每个节点的最大边数
                "efConstruction": 200,  # 构建时的搜索宽度
            },
        ),
        # 标量字段索引（用于过滤）
        IndexConfig(
            field_name="user_id", index_type="AUTOINDEX"  # 自动选择最适合的索引类型
        ),
        IndexConfig(field_name="group_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="event_type", index_type="AUTOINDEX"),
        IndexConfig(field_name="timestamp", index_type="AUTOINDEX"),
    ]
