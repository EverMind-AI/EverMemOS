"""
语义记忆 Milvus Collection 定义

基于 MilvusCollectionWithSuffix 实现的语义记忆专用 Collection 类。
提供了与 SemanticMemoryMilvusRepository 兼容的 Schema 定义和索引配置。
支持个人语义记忆和群组语义记忆。
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import (
    MilvusCollectionWithSuffix,
    IndexConfig,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class SemanticMemoryCollection(MilvusCollectionWithSuffix):
    """
    语义记忆 Milvus Collection

    同时支持个人语义记忆和群组语义记忆，通过 group_id 字段区分。

    使用方式：
        # 使用 Collection
        collection.async_collection().insert([...])
        collection.async_collection().search([...])
    """

    # Collection 基础名称
    _COLLECTION_NAME = "semantic_memory"

    # Collection Schema 定义
    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="语义记忆唯一标识",
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
                description="相关参与者列表",
            ),
            FieldSchema(
                name="parent_episode_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="父情景记忆ID",
            ),
            FieldSchema(
                name="start_time",
                dtype=DataType.INT64,
                description="语义记忆开始时间戳",
            ),
            FieldSchema(
                name="end_time", dtype=DataType.INT64, description="语义记忆结束时间戳"
            ),
            FieldSchema(
                name="duration_days", dtype=DataType.INT64, description="持续天数"
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="语义记忆内容",
            ),
            FieldSchema(
                name="evidence",
                dtype=DataType.VARCHAR,
                max_length=2000,
                description="支持该语义记忆的证据",
            ),
            FieldSchema(
                name="search_content",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="搜索内容（JSON格式）",
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=50000,
                description="详细信息JSON（元数据）",
            ),
            FieldSchema(
                name="created_at", dtype=DataType.INT64, description="创建时间戳"
            ),
            FieldSchema(
                name="updated_at", dtype=DataType.INT64, description="更新时间戳"
            ),
        ],
        description="个人语义记忆向量集合",
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
        IndexConfig(field_name="parent_episode_id", index_type="AUTOINDEX"),
        IndexConfig(field_name="start_time", index_type="AUTOINDEX"),
        IndexConfig(field_name="end_time", index_type="AUTOINDEX"),
    ]
