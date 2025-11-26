"""
事件日志 Milvus 仓库

基于 BaseMilvusRepository 的事件日志专用仓库类，提供高效的向量存储和检索功能。
主要功能包括向量存储、相似性搜索、过滤查询和文档管理。
支持个人和群组事件日志。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import json
from core.oxm.milvus.base_repository import BaseMilvusRepository
from infra_layer.adapters.out.search.milvus.memory.event_log_collection import (
    EventLogCollection,
)
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("event_log_milvus_repository", primary=False)
class EventLogMilvusRepository(BaseMilvusRepository[EventLogCollection]):
    """
    事件日志 Milvus 仓库

    基于 BaseMilvusRepository 的专用仓库类，提供：
    - 高效的向量存储和检索
    - 相似性搜索和过滤功能
    - 文档创建和管理
    - 向量索引管理

    同时支持个人和群组事件日志。
    """

    def __init__(self):
        """初始化事件日志仓库"""
        super().__init__(EventLogCollection)

    # ==================== 文档创建和管理 ====================

    async def create_and_save_event_log(
        self,
        id: str,
        user_id: Optional[str],
        atomic_fact: str,
        parent_episode_id: str,
        timestamp: datetime,
        vector: List[float],
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        search_content: Optional[List[str]] = None,
        extend: Optional[Dict[str, Any]] = None,
        vector_model: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        创建并保存事件日志文档

        Args:
            id: 事件日志唯一标识
            user_id: 用户ID（必需）
            atomic_fact: 原子事实内容（必需）
            parent_episode_id: 父情景记忆ID（必需）
            timestamp: 事件发生时间（必需）
            vector: 文本向量（必需，维度必须为VECTORIZE_DIMENSIONS）
            group_id: 群组ID
            participants: 相关参与者列表
            event_type: 事件类型（如 Conversation, Email 等）
            search_content: 搜索内容列表
            extend: 扩展字段
            vector_model: 向量化模型
            created_at: 创建时间
            updated_at: 更新时间

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

            # 构建搜索内容
            if search_content is None:
                search_content = [atomic_fact]

            # 准备元数据
            metadata = {"vector_model": vector_model or "", "extend": extend or {}}

            # 准备实体数据
            entity = {
                "id": id,
                "vector": vector,
                "user_id": user_id or "",
                "group_id": group_id or "",
                "participants": participants or [],
                "parent_episode_id": parent_episode_id,
                "event_type": event_type,
                "timestamp": int(timestamp.timestamp()),
                "atomic_fact": atomic_fact,
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "created_at": int(created_at.timestamp()),
                "updated_at": int(updated_at.timestamp()),
            }

            # 插入数据
            await self.insert(entity)

            logger.debug("✅ 创建事件日志文档成功: id=%s, user_id=%s", id, user_id)

            return {
                "id": id,
                "user_id": user_id,
                "atomic_fact": atomic_fact,
                "parent_episode_id": parent_episode_id,
                "timestamp": timestamp,
                "search_content": search_content,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error("❌ 创建事件日志文档失败: id=%s, error=%s", id, e)
            raise

    # ==================== 搜索功能 ====================

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        parent_episode_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
        participant_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似性搜索

        Args:
            query_vector: 查询向量
            user_id: 用户ID过滤
            group_id: 群组ID过滤
            parent_episode_id: 父情景记忆ID过滤
            event_type: 事件类型过滤
            start_time: 事件时间戳开始过滤
            end_time: 事件时间戳结束过滤
            limit: 返回结果数量
            score_threshold: 相似度阈值
            participant_user_id: 群组检索时额外要求参与者包含该用户

        Returns:
            搜索结果列表
        """
        try:
            # 构建过滤表达式
            filter_expr = []

            if user_id:
                filter_expr.append(f'user_id == "{user_id}"')
            else:
                # 如果没有传 user_id,则过滤所有非空 user_id
                filter_expr.append('user_id == ""')
            if group_id:
                filter_expr.append(f'group_id == "{group_id}"')

            if participant_user_id:
                filter_expr.append(
                    f'array_contains(participants, "{participant_user_id}")'
                )
            if group_id:
                filter_expr.append(f'group_id == "{group_id}"')
            if parent_episode_id:
                filter_expr.append(f'parent_episode_id == "{parent_episode_id}"')
            if event_type:
                filter_expr.append(f'event_type == "{event_type}"')
            if start_time:
                filter_expr.append(f"timestamp >= {int(start_time.timestamp())}")
            if end_time:
                filter_expr.append(f"timestamp <= {int(end_time.timestamp())}")

            filter_str = " and ".join(filter_expr) if filter_expr else None

            similarity_threshold: Optional[float] = (
                radius if radius is not None else None
            )

            # 执行搜索
            # 动态调整 ef 参数：必须 >= limit，通常设为 limit 的 1.5-2 倍
            ef_value = max(128, limit * 2)  # 确保 ef >= limit，至少 128
            search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}

            # 不设置 radius 参数!
            # Milvus 的 radius 是相似度下限,设置 -1.0 反而可能导致问题
            # 我们通过后置过滤来控制相似度阈值
            if radius is not None and radius > -1.0:
                search_params["params"]["radius"] = radius

            logger.info(
                f"Milvus 搜索参数: limit={limit}, "
                f"radius={search_params['params'].get('radius', 'None')}, "
                f"filter_str={filter_str}"
            )

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
            raw_hit_count = sum(len(hits) for hits in results)
            logger.info(
                f"Milvus 原始返回: {raw_hit_count} 条结果, "
                f"limit={limit}, filter_str={filter_str}, "
            )

            for hits in results:
                for hit in hits:
                    threshold = (
                        similarity_threshold
                        if similarity_threshold is not None
                        else score_threshold
                    )
                    keep = hit.score >= threshold

                    if keep:
                        # 解析元数据
                        metadata_json = hit.entity.get("metadata", "{}")
                        metadata = json.loads(metadata_json) if metadata_json else {}

                        # 解析 search_content（统一为 JSON 数组格式）
                        search_content_raw = hit.entity.get("search_content", "[]")
                        search_content = (
                            json.loads(search_content_raw) if search_content_raw else []
                        )

                        # 构建结果
                        result = {
                            "id": hit.entity.get("id"),
                            "score": float(hit.score),
                            "user_id": hit.entity.get("user_id"),
                            "group_id": hit.entity.get("group_id"),
                            "parent_episode_id": hit.entity.get("parent_episode_id"),
                            "event_type": hit.entity.get("event_type"),
                            "timestamp": datetime.fromtimestamp(
                                hit.entity.get("timestamp", 0)
                            ),
                            "atomic_fact": hit.entity.get("atomic_fact"),
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

    async def delete_by_id(self, log_id: str) -> bool:
        """
        根据log_id删除事件日志文档

        Args:
            log_id: 事件日志唯一标识

        Returns:
            删除成功返回 True，否则返回 False
        """
        try:
            success = await super().delete_by_id(log_id)
            if success:
                logger.debug("✅ 根据log_id删除事件日志成功: log_id=%s", log_id)
            return success
        except Exception as e:
            logger.error(
                "❌ 根据log_id删除事件日志失败: log_id=%s, error=%s", log_id, e
            )
            return False

    async def delete_by_parent_episode_id(self, parent_episode_id: str) -> int:
        """
        根据父情景记忆ID删除所有关联的事件日志

        Args:
            parent_episode_id: 父情景记忆ID

        Returns:
            删除的文档数量
        """
        try:
            expr = f'parent_episode_id == "{parent_episode_id}"'

            # 先查询要删除的文档数量
            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            if delete_count > 0:
                # 执行删除
                await self.collection.delete(expr)

            logger.debug(
                "✅ 根据parent_episode_id删除事件日志成功: 删除了 %d 条记录",
                delete_count,
            )
            return delete_count

        except Exception as e:
            logger.error("❌ 根据parent_episode_id删除事件日志失败: %s", e)
            raise

    async def delete_by_filters(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        根据过滤条件批量删除事件日志文档

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
            if user_id is not None:  # 使用 is not None 而不是 truthy 检查，支持空字符串
                if user_id:  # 非空字符串：个人记忆
                    # 同时检查 user_id 字段和 participants 数组
                    user_filter = f'(user_id == "{user_id}" or array_contains(participants, "{user_id}"))'
                    filter_expr.append(user_filter)
                else:  # 空字符串：群组记忆
                    filter_expr.append('user_id == ""')
            if group_id:
                filter_expr.append(f'group_id == "{group_id}"')
            if start_time:
                filter_expr.append(f"timestamp >= {int(start_time.timestamp())}")
            if end_time:
                filter_expr.append(f"timestamp <= {int(end_time.timestamp())}")

            if not filter_expr:
                raise ValueError("至少需要提供一个过滤条件")

            expr = " and ".join(filter_expr)

            # 先查询要删除的文档数量
            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            # 执行删除
            await self.collection.delete(expr)

            logger.debug(
                "✅ 根据过滤条件批量删除事件日志成功: 删除了 %d 条记录", delete_count
            )
            return delete_count

        except Exception as e:
            logger.error("❌ 根据过滤条件批量删除事件日志失败: %s", e)
            raise
