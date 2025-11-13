"""
情景记忆 Milvus 仓库

基于 BaseMilvusRepository 的情景记忆专用仓库类，提供高效的向量存储和检索功能。
主要功能包括向量存储、相似性搜索、过滤查询和文档管理。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import json
from core.oxm.milvus.base_repository import BaseMilvusRepository
from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone
from core.di.decorators import repository

logger = get_logger(__name__)

# Milvus 检索配置
MILVUS_SIMILARITY_RADIUS = 0.6  # COSINE 相似度阈值，范围 [-1, 1]，1 表示完全相同


@repository("episodic_memory_milvus_repository", primary=False)
class EpisodicMemoryMilvusRepository(BaseMilvusRepository[EpisodicMemoryCollection]):
    """
    情景记忆 Milvus 仓库

    基于 BaseMilvusRepository 的专用仓库类，提供：
    - 高效的向量存储和检索
    - 相似性搜索和过滤功能
    - 文档创建和管理
    - 向量索引管理
    """

    def __init__(self):
        """初始化情景记忆仓库"""
        super().__init__(EpisodicMemoryCollection)

    # ==================== 文档创建和管理 ====================

    async def create_and_save_episodic_memory(
        self,
        event_id: str,
        user_id: str,
        timestamp: datetime,
        episode: str,
        search_content: List[str],
        vector: List[float],
        memory_sub_type: str = "episode",
        user_name: Optional[str] = None,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        linked_entities: Optional[List[str]] = None,
        subject: Optional[str] = None,
        memcell_event_id_list: Optional[List[str]] = None,
        extend: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        parent_event_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        创建并保存情景记忆文档

        Args:
            event_id: 事件唯一标识
            user_id: 用户ID（必需）
            timestamp: 事件发生时间（必需）
            episode: 情景描述（必需）
            search_content: 搜索内容列表（必需）
            vector: 文本向量（必需，维度必须为1024）
            memory_sub_type: 记忆子类型，可选值：episode/semantic_memory/event_log（默认 episode）
            user_name: 用户名称
            title: 事件标题
            summary: 事件摘要
            group_id: 群组ID
            participants: 参与者列表
            event_type: 事件类型（如 conversation, email 等）
            keywords: 关键词列表
            linked_entities: 关联实体ID列表
            subject: 事件标题
            memcell_event_id_list: 记忆单元事件ID列表
            extend: 扩展字段
            created_at: 创建时间
            updated_at: 更新时间
            parent_event_id: 父事件ID（用于关联拆分的记录）
            start_time: 语义记忆开始时间（仅 semantic_memory 类型使用）
            end_time: 语义记忆结束时间（仅 semantic_memory 类型使用）

        Returns:
            已保存的文档信息
        """
        try:
            # 设置默认时间戳
            now = get_now_with_timezone()
            if created_at is None:
                created_at = now
            if updated_at is None:
                updated_at = now

            # 准备元数据
            metadata = {
                "user_name": user_name or "",
                "title": title or "",
                "summary": summary or "",
                "participants": participants or [],
                "keywords": keywords or [],
                "linked_entities": linked_entities or [],
                "subject": subject or "",
                "memcell_event_id_list": memcell_event_id_list or [],
                "extend": extend or {},
                "created_at": created_at.isoformat(),
                "updated_at": updated_at.isoformat(),
                "parent_event_id": parent_event_id or "",  # 父事件ID，用于关联拆分的记录
            }

            # 准备实体数据
            entity = {
                "id": event_id,
                "vector": vector,
                "user_id": user_id,
                "group_id": group_id or "",
                "participants": participants or [],  # 添加 participants 字段
                "event_type": event_type or "",
                "memory_sub_type": memory_sub_type,  # 记忆子类型
                "timestamp": int(timestamp.timestamp()),
                "start_time": int(start_time.timestamp()) if start_time else 0,
                "end_time": int(end_time.timestamp()) if end_time else 0,
                "episode": episode,
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "created_at": int(created_at.timestamp()),
                "updated_at": int(updated_at.timestamp()),
            }

            # 插入数据
            await self.insert(entity)

            logger.debug(
                "✅ 创建情景记忆文档成功: event_id=%s, user_id=%s", event_id, user_id
            )

            return {
                "event_id": event_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "episode": episode,
                "search_content": search_content,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error("❌ 创建情景记忆文档失败: event_id=%s, error=%s", event_id, e)
            raise

    # ==================== 搜索功能 ====================

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        event_type: Optional[str] = None,
        memory_sub_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        semantic_start_time: Optional[datetime] = None,
        semantic_end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似性搜索

        Args:
            query_vector: 查询向量
            user_id: 用户ID过滤
            group_id: 群组ID过滤
            event_type: 事件类型过滤（如 conversation, email 等）
            memory_sub_type: 记忆子类型过滤（episode/semantic_memory/event_log）
            start_time: 事件时间戳开始过滤
            end_time: 事件时间戳结束过滤
            semantic_start_time: 语义记忆有效期开始时间过滤（仅对 semantic_memory 有效）
            semantic_end_time: 语义记忆有效期结束时间过滤（仅对 semantic_memory 有效）
            radius: COSINE 相似度阈值（可选，默认使用 MILVUS_SIMILARITY_RADIUS）
            limit: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            搜索结果列表
        """
        try:
            # 构建过滤表达式
            filter_expr = []
            if user_id:
                # 同时检查 user_id 字段和 participants 数组
                # 使用 OR 逻辑：user_id 匹配 或者 user_id 在 participants 中
                user_filter = f'(user_id == "{user_id}" or array_contains(participants, "{user_id}"))'
                filter_expr.append(user_filter)
            if group_id:
                filter_expr.append(f'group_id == "{group_id}"')
            if event_type:
                filter_expr.append(f'event_type == "{event_type}"')
            if memory_sub_type:
                # 支持模糊匹配：如果 memory_sub_type 是 "episode"/"event_log"/"semantic_memory"
                # 则匹配包含该子串的所有类型（例如 "episode" 匹配 "episode" 和 "personal_episode"）
                if memory_sub_type in ["episode", "event_log", "semantic_memory"]:
                    # 使用 like 操作符进行模糊匹配
                    filter_expr.append(f'memory_sub_type like "%{memory_sub_type}%"')
                else:
                    # 精确匹配（例如 "personal_episode", "personal_event_log"）
                    filter_expr.append(f'memory_sub_type == "{memory_sub_type}"')
            if start_time:
                filter_expr.append(f'timestamp >= {int(start_time.timestamp())}')
            if end_time:
                filter_expr.append(f'timestamp <= {int(end_time.timestamp())}')
            if semantic_start_time:
                filter_expr.append(f'start_time >= {int(semantic_start_time.timestamp())}')
            if semantic_end_time:
                filter_expr.append(f'end_time <= {int(semantic_end_time.timestamp())}')

            filter_str = " and ".join(filter_expr) if filter_expr else None

            # 获取集合

            # 执行搜索
            # 动态调整 ef 参数：必须 >= limit，通常设为 limit 的 1.5-2 倍
            ef_value = max(128, limit * 2)  # 确保 ef >= limit，至少 128
            # 使用 COSINE 相似度，radius 表示只返回相似度 >= 阈值的结果
            # 优先使用传入的 radius 参数，否则使用默认配置
            similarity_radius = radius if radius is not None else MILVUS_SIMILARITY_RADIUS
            search_params = {
                "metric_type": "COSINE",
                "params": {
                    "ef": ef_value,
                    "radius": similarity_radius,  # 相似度阈值（COSINE 范围 -1 到 1，1 表示完全相同）
                }
            }

            results = await self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filter_str,
                output_fields=self.all_output_fields,
            )

            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        # 解析元数据
                        metadata_json = hit.entity.get("metadata", "{}")
                        metadata = json.loads(metadata_json) if metadata_json else {}

                        # 解析 search_content（统一为 JSON 数组格式）
                        search_content_raw = hit.entity.get("search_content", "[]")
                        search_content = json.loads(search_content_raw) if search_content_raw else []

                        result = {
                            "id": hit.entity.get("id"),
                            "score": float(hit.score),
                            "user_id": hit.entity.get("user_id"),
                            "group_id": hit.entity.get("group_id"),
                            "event_type": hit.entity.get("event_type"),
                            "memory_sub_type": hit.entity.get("memory_sub_type"),
                            "timestamp": datetime.fromtimestamp(
                                hit.entity.get("timestamp", 0)
                            ),
                            "episode": hit.entity.get("episode"),
                            "search_content": search_content,
                            "metadata": metadata,
                        }
                        search_results.append(result)

            logger.debug("✅ 向量搜索成功: 找到 %d 条结果", len(search_results))
            return search_results

        except Exception as e:
            logger.error("❌ 向量搜索失败: %s", e)
            raise

    # ==================== 删除功能 ====================

    async def delete_by_event_id(self, event_id: str) -> bool:
        """
        根据event_id删除情景记忆文档

        Args:
            event_id: 事件唯一标识

        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            success = await self.delete_by_id(event_id)
            if success:
                logger.debug("✅ 根据event_id删除情景记忆成功: event_id=%s", event_id)
            return success
        except Exception as e:
            logger.error(
                "❌ 根据event_id删除情景记忆失败: event_id=%s, error=%s", event_id, e
            )
            return False

    async def delete_by_filters(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        根据过滤条件批量删除情景记忆文档

        Args:
            user_id: 用户ID过滤
            group_id: 群组ID过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            删除的文档数量
        """
        try:
            # 构建过滤表达式
            filter_expr = []
            if user_id:
                # 同时检查 user_id 字段和 participants 数组
                user_filter = f'(user_id == "{user_id}" or array_contains(participants, "{user_id}"))'
                filter_expr.append(user_filter)
            if group_id:
                filter_expr.append(f'group_id == "{group_id}"')
            if start_time:
                filter_expr.append(f'timestamp >= {int(start_time.timestamp())}')
            if end_time:
                filter_expr.append(f'timestamp <= {int(end_time.timestamp())}')

            if not filter_expr:
                raise ValueError("至少需要提供一个过滤条件")

            expr = " and ".join(filter_expr)

            # 先查询要删除的文档数量
            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            # 执行删除
            await self.collection.delete(expr)

            logger.debug(
                "✅ 根据过滤条件批量删除情景记忆成功: 删除了 %d 条记录", delete_count
            )
            return delete_count

        except Exception as e:
            logger.error("❌ 根据过滤条件批量删除情景记忆失败: %s", e)
            raise
