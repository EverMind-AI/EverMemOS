"""
事件日志 Milvus Collection 定义

基于 MilvusCollectionWithSuffix 实现的事件日志专用 Collection 类。
提供了与 EventLogMilvusRepository 兼容的 Schema 定义和索引配置。
支持个人和群组事件日志。
"""

from pymilvus import DataType, FieldSchema, CollectionSchema
from core.oxm.milvus.milvus_collection_base import IndexConfig
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection_with_suffix import (
    TenantAwareMilvusCollectionWithSuffix,
)
from memory_layer.constants import VECTORIZE_DIMENSIONS


class EventLogCollection(TenantAwareMilvusCollectionWithSuffix):
    """
    事件日志 Milvus Collection

    存储原子事实（atomic facts），支持细粒度的事实检索。
    同时支持个人和群组事件日志，通过 group_id 字段区分。

    使用方式：
        # 使用 Collection
        collection.async_collection().insert([...])
        collection.async_collection().search([...])
    """

    # Collection 基础名称
    _COLLECTION_NAME = "event_log"

    # Collection Schema 定义
    _SCHEMA = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
                description="事件日志唯一标识",
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
                name="event_type",
                dtype=DataType.VARCHAR,
                max_length=50,
                description="事件类型（如 Conversation, Email 等）",
            ),
            FieldSchema(
                name="timestamp", dtype=DataType.INT64, description="事件发生时间戳"
            ),
            FieldSchema(
                name="atomic_fact",
                dtype=DataType.VARCHAR,
                max_length=5000,
                description="原子事实内容",
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
        description="事件日志向量集合",
        enable_dynamic_field=True,
    )

    # 索引配置
    _INDEX_CONFIGS = [
        # 向量字段索引（用于相似度搜索）
        IndexConfig(
            field_name="vector",
            index_type="HNSW",  # 高效的近似最近邻搜索
            metric_type="COSINE",  # 统一使用余弦相似度
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
        IndexConfig(field_name="event_type", index_type="AUTOINDEX"),
        IndexConfig(field_name="timestamp", index_type="AUTOINDEX"),
    ]
