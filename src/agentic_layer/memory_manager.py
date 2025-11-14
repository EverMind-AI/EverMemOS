from __future__ import annotations

from typing import Any, List, Optional
import logging
import asyncio

from datetime import datetime, timedelta
import jieba
import numpy as np
import time
from typing import Dict, Any
from dataclasses import dataclass

from memory_layer.types import Memory
from biz_layer.mem_memorize import memorize
from memory_layer.memory_manager import MemorizeRequest
from .fetch_mem_service import get_fetch_memory_service
from .dtos.memory_query import (
    FetchMemRequest,
    FetchMemResponse,
    RetrieveMemRequest,
    RetrieveMemResponse,
    Metadata,
)
from core.di import get_bean_by_type
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from core.observation.tracing.decorators import trace_logger
from core.nlp.stopwords_utils import filter_stopwords
from common_utils.datetime_utils import from_iso_format, get_now_with_timezone
from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
    MemCellRawRepository,
)
from infra_layer.adapters.out.persistence.document.memory.user_profile import (
    UserProfile,
)
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from .vectorize_service import get_vectorize_service
from .rerank_service import get_rerank_service
from .retrieval_utils import lightweight_retrieval, build_bm25_index, search_with_bm25
import os

logger = logging.getLogger(__name__)


@dataclass
class EventLogCandidate:
    """Event Log 候选对象（用于从 atomic_fact 检索）"""
    event_id: str
    user_id: str
    group_id: str
    timestamp: datetime
    episode: str  # atomic_fact 内容
    summary: str
    subject: str
    extend: dict  # 包含 embedding


class MemoryManager:
    """Unified memory interface.

    提供以下主要功能:
    - memorize: 接受原始数据并持久化存储
    - fetch_mem: 通过键检索记忆字段，支持多种记忆类型
    - retrieve_mem: 基于提示词检索方法的记忆读取
    """

    def __init__(self) -> None:
        # 获取记忆服务实例
        self._fetch_service = get_fetch_memory_service()

        logger.info(
            "MemoryManager initialized with fetch_mem_service and retrieve_mem_service"
        )

    # --------- Write path (raw data -> memorize) ---------
    @trace_logger(operation_name="agentic_layer 记忆存储")
    async def memorize(self, memorize_request: MemorizeRequest) -> List[Memory]:
        """Memorize a heterogeneous list of raw items.

        Accepts list[Any], where each item can be one of the typed raw dataclasses
        (ChatRawData / EmailRawData / MemoRawData / LincDocRawData) or any dict-like
        object. Each item is stored as a MemoryCell with a synthetic key.
        """
        memories = await memorize(memorize_request)
        return memories

    # --------- Read path (query -> fetch_mem) ---------
    # 基于kv的记忆读取，包括静态与动态记忆
    @trace_logger(operation_name="agentic_layer 记忆读取")
    async def fetch_mem(self, request: FetchMemRequest) -> FetchMemResponse:
        """获取记忆数据，支持多种记忆类型

        Args:
            request: FetchMemRequest 包含查询参数

        Returns:
            FetchMemResponse 包含查询结果
        """
        logger.debug(
            f"fetch_mem called with request: user_id={request.user_id}, memory_type={request.memory_type}"
        )

        # repository 支持 MemoryType.MULTIPLE 类型，默认就是corememory
        response = await self._fetch_service.find_by_user_id(
            user_id=request.user_id,
            memory_type=request.memory_type,
            version_range=request.version_range,
            limit=request.limit,
        )

        # 注意：response.metadata 已经通过 _get_employee_metadata 包含了完整的员工信息
        # 包括 source, user_id, memory_type, limit, email, phone, full_name
        # 这里不需要再次更新，因为 fetch_mem_service 已经提供了正确的信息

        logger.debug(
            f"fetch_mem returned {len(response.memories)} memories for user {request.user_id}"
        )
        return response

    # 基于retrieve_method的记忆读取，包括静态与动态记忆
    @trace_logger(operation_name="agentic_layer 记忆检索")
    async def retrieve_mem(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """检索记忆数据，根据 retrieve_method 分发到不同的检索方法

        Args:
            retrieve_mem_request: RetrieveMemRequest 包含检索参数

        Returns:
            RetrieveMemResponse 包含检索结果
        """
        try:
            # 验证请求参数
            if not retrieve_mem_request:
                raise ValueError("retrieve_mem_request is required for retrieve_mem")

            # 根据 retrieve_method 分发到不同的检索方法
            from .memory_models import RetrieveMethod

            retrieve_method = retrieve_mem_request.retrieve_method

            logger.info(
                f"retrieve_mem 分发请求: user_id={retrieve_mem_request.user_id}, "
                f"retrieve_method={retrieve_method}, query={retrieve_mem_request.query}"
            )

            # 根据检索方法分发
            if retrieve_method == RetrieveMethod.KEYWORD:
                # 关键词检索
                return await self.retrieve_mem_keyword(retrieve_mem_request)
            elif retrieve_method == RetrieveMethod.VECTOR:
                # 向量检索
                return await self.retrieve_mem_vector(retrieve_mem_request)
            elif retrieve_method == RetrieveMethod.HYBRID:
                # 混合检索
                return await self.retrieve_mem_hybrid(retrieve_mem_request)
            else:
                raise ValueError(f"不支持的检索方法: {retrieve_method}")

        except Exception as e:
            logger.error(f"Error in retrieve_mem: {e}", exc_info=True)
            return RetrieveMemResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve",
                ),
                metadata=Metadata(
                    source="retrieve_mem_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve",
                ),
            )

    # 关键词检索方法（原来的 retrieve_mem 逻辑）
    @trace_logger(operation_name="agentic_layer 关键词记忆检索")
    async def retrieve_mem_keyword(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """基于关键词的记忆检索（原 retrieve_mem 的实现）

        Args:
            retrieve_mem_request: RetrieveMemRequest 包含检索参数

        Returns:
            RetrieveMemResponse 包含检索结果
        """
        try:
            # 从 Request 中获取参数
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for retrieve_mem_keyword"
                )

            search_results = await self.get_keyword_search_results(retrieve_mem_request)

            if not search_results:
                logger.warning(
                    f"关键词检索未找到结果: user_id={retrieve_mem_request.user_id}, query={retrieve_mem_request.query}"
                )
                return RetrieveMemResponse(
                    memories=[],
                    original_data=[],
                    scores=[],
                    importance_scores=[],
                    total_count=0,
                    has_more=False,
                    query_metadata=Metadata(
                        source="episodic_memory_es_repository",
                        user_id=retrieve_mem_request.user_id,
                        memory_type="retrieve_keyword",
                    ),
                    metadata=Metadata(
                        source="episodic_memory_es_repository",
                        user_id=retrieve_mem_request.user_id,
                        memory_type="retrieve_keyword",
                    ),
                )

            # 使用通用的分组处理策略
            memories, scores, importance_scores, original_data, total_count = (
                await self.group_by_groupid_stratagy(search_results, source_type="es")
            )

            logger.debug(
                f"EpisodicMemoryEsRepository multi_search returned {len(memories)} groups for query: {retrieve_mem_request.query}"
            )

            return RetrieveMemResponse(
                memories=memories,
                scores=scores,
                importance_scores=importance_scores,
                original_data=original_data,
                total_count=total_count,
                has_more=False,
                query_metadata=Metadata(
                    source="episodic_memory_es_repository",
                    user_id=retrieve_mem_request.user_id,
                    memory_type="retrieve_keyword",
                ),
                metadata=Metadata(
                    source="episodic_memory_es_repository",
                    user_id=retrieve_mem_request.user_id,
                    memory_type="retrieve_keyword",
                ),
            )

        except Exception as e:
            logger.error(f"Error in retrieve_mem_keyword: {e}", exc_info=True)
            return RetrieveMemResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_keyword_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve_keyword",
                ),
                metadata=Metadata(
                    source="retrieve_mem_keyword_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve_keyword",
                ),
            )

    async def get_keyword_search_results(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> Dict[str, Any]:
        try:
            # 从 Request 中获取参数
            if not retrieve_mem_request:
                raise ValueError("retrieve_mem_request is required for retrieve_mem")

            top_k = retrieve_mem_request.top_k
            query = retrieve_mem_request.query
            user_id = retrieve_mem_request.user_id
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            # 获取 EpisodicMemoryEsRepository 实例
            es_repo = get_bean_by_type(EpisodicMemoryEsRepository)

            # 将查询字符串转换为搜索词列表
            # 使用jieba进行搜索模式分词，然后过滤停用词
            if query:
                raw_words = list(jieba.cut_for_search(query))
                query_words = filter_stopwords(raw_words, min_length=2)
            else:
                query_words = []

            logger.debug(f"query_words: {query_words}")

            # 构建时间范围过滤条件，处理 None 值
            date_range = {}
            if start_time is not None:
                date_range["gte"] = start_time
            if end_time is not None:
                date_range["lte"] = end_time

            # 调用 multi_search 方法，支持按 memory_sub_type 过滤
            search_results = await es_repo.multi_search(
                query=query_words,
                user_id=user_id,
                event_type=retrieve_mem_request.memory_sub_type,  # 按记忆子类型过滤
                size=top_k,
                from_=0,
                date_range=date_range,
            )
            return search_results
        except Exception as e:
            logger.error(f"Error in get_keyword_search_results: {e}")
            return {}

    # 基于向量的记忆检索
    @trace_logger(operation_name="agentic_layer 向量记忆检索")
    async def retrieve_mem_vector(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """基于向量相似性的记忆检索

        Args:
            request: Request 包含检索参数，包括 query 和 retrieve_mem_request

        Returns:
            RetrieveMemResponse 包含检索结果
        """
        try:
            # 从 Request 中获取参数
            logger.debug(
                f"retrieve_mem_vector called with retrieve_mem_request: {retrieve_mem_request}"
            )
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for retrieve_mem_vector"
                )

            query = retrieve_mem_request.query
            if not query:
                raise ValueError("query is required for retrieve_mem_vector")

            user_id = retrieve_mem_request.user_id
            top_k = retrieve_mem_request.top_k
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            logger.debug(
                f"retrieve_mem_vector called with query: {query}, user_id: {user_id}, top_k: {top_k}"
            )

            # 获取向量化服务
            vectorize_service = get_vectorize_service()

            # 将查询文本转换为向量
            logger.debug(f"开始向量化查询文本: {query}")
            query_vector = await vectorize_service.get_embedding(query)
            query_vector_list = query_vector.tolist()  # 转换为列表格式
            logger.debug(f"查询文本向量化完成，向量维度: {len(query_vector_list)}")

            # 根据 memory_sub_type 选择对应的 Milvus Repository
            # - "semantic_memory": 使用 SemanticMemoryMilvusRepository
            # - "event_log": 使用 EventLogMilvusRepository
            # - 其他（episode/None）: 使用 EpisodicMemoryMilvusRepository（默认）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                from infra_layer.adapters.out.search.repository.semantic_memory_milvus_repository import (
                    SemanticMemoryMilvusRepository,
                )
                milvus_repo = get_bean_by_type(SemanticMemoryMilvusRepository)
            elif retrieve_mem_request.memory_sub_type == "event_log":
                from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
                    EventLogMilvusRepository,
                )
                milvus_repo = get_bean_by_type(EventLogMilvusRepository)
            else:
                milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)

            # 处理时间范围过滤条件
            start_time_dt = None
            end_time_dt = None
            semantic_start_dt = None
            semantic_end_dt = None
            current_time_dt = None

            if start_time is not None:
                if isinstance(start_time, str):
                    # 如果是日期格式 "2024-01-01"，转换为当天的开始时间
                    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d")
                else:
                    start_time_dt = start_time

            if end_time is not None:
                if isinstance(end_time, str):
                    # 如果是日期格式 "2024-12-31"，转换为当天的结束时间
                    end_time_dt = datetime.strptime(end_time, "%Y-%m-%d")
                    # 设置为当天的23:59:59，确保包含整天
                    end_time_dt = end_time_dt.replace(hour=23, minute=59, second=59)
                else:
                    end_time_dt = end_time

            # 处理语义记忆时间范围（仅对 semantic_memory 有效）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                if retrieve_mem_request.semantic_start_time:
                    semantic_start_dt = datetime.strptime(retrieve_mem_request.semantic_start_time, "%Y-%m-%d")
                if retrieve_mem_request.semantic_end_time:
                    semantic_end_dt = datetime.strptime(retrieve_mem_request.semantic_end_time, "%Y-%m-%d")
                if retrieve_mem_request.current_time:
                    current_time_dt = datetime.strptime(retrieve_mem_request.current_time, "%Y-%m-%d")

            # 调用 Milvus 的向量搜索（根据记忆类型传递不同的参数）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                # 语义记忆：支持时间范围和有效期过滤
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=semantic_start_dt,
                    end_time=semantic_end_dt,
                    current_time=current_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                )
            else:
                # 情景记忆和事件日志：使用 timestamp 过滤
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=start_time_dt,
                    end_time=end_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                )

            logger.debug(f"Milvus向量搜索返回 {len(search_results)} 条结果")

            # 使用通用的分组处理策略
            memories, scores, importance_scores, original_data, total_count = (
                await self.group_by_groupid_stratagy(
                    search_results, source_type="milvus"
                )
            )

            logger.debug(
                f"EpisodicMemoryMilvusRepository vector_search returned {len(memories)} groups for query: {query}"
            )

            return RetrieveMemResponse(
                memories=memories,
                scores=scores,
                importance_scores=importance_scores,
                original_data=original_data,
                total_count=total_count,
                has_more=False,
                query_metadata=Metadata(
                    source="episodic_memory_milvus_repository",
                    user_id=user_id,
                    memory_type="retrieve_vector",
                ),
                metadata=Metadata(
                    source="episodic_memory_milvus_repository",
                    user_id=user_id,
                    memory_type="retrieve_vector",
                ),
            )

        except Exception as e:
            logger.error(f"Error in retrieve_mem_vector: {e}")
            return RetrieveMemResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_vector_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_vector",
                ),
                metadata=Metadata(
                    source="retrieve_mem_vector_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_vector",
                ),
            )

    async def get_vector_search_results(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> Dict[str, Any]:
        try:
            # 从 Request 中获取参数
            logger.debug(
                f"get_vector_search_results called with retrieve_mem_request: {retrieve_mem_request}"
            )
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for get_vector_search_results"
                )
            query = retrieve_mem_request.query
            if not query:
                raise ValueError("query is required for retrieve_mem_vector")

            user_id = retrieve_mem_request.user_id
            top_k = retrieve_mem_request.top_k
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            logger.debug(
                f"retrieve_mem_vector called with query: {query}, user_id: {user_id}, top_k: {top_k}"
            )

            # 获取向量化服务
            vectorize_service = get_vectorize_service()

            # 将查询文本转换为向量
            logger.debug(f"开始向量化查询文本: {query}")
            query_vector = await vectorize_service.get_embedding(query)
            query_vector_list = query_vector.tolist()  # 转换为列表格式
            logger.debug(f"查询文本向量化完成，向量维度: {len(query_vector_list)}")

            # 根据 memory_sub_type 选择对应的 Milvus Repository
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                from infra_layer.adapters.out.search.repository.semantic_memory_milvus_repository import (
                    SemanticMemoryMilvusRepository,
                )
                milvus_repo = get_bean_by_type(SemanticMemoryMilvusRepository)
            elif retrieve_mem_request.memory_sub_type == "event_log":
                from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
                    EventLogMilvusRepository,
                )
                milvus_repo = get_bean_by_type(EventLogMilvusRepository)
            else:
                milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)

            # 处理时间范围过滤条件
            start_time_dt = None
            end_time_dt = None
            semantic_start_dt = None
            semantic_end_dt = None
            current_time_dt = None

            if start_time is not None:
                if isinstance(start_time, str):
                    # 如果是日期格式 "2024-01-01"，转换为当天的开始时间
                    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d")
                else:
                    start_time_dt = start_time

            if end_time is not None:
                if isinstance(end_time, str):
                    # 如果是日期格式 "2024-12-31"，转换为当天的结束时间
                    end_time_dt = datetime.strptime(end_time, "%Y-%m-%d")
                    # 设置为当天的23:59:59，确保包含整天
                    end_time_dt = end_time_dt.replace(hour=23, minute=59, second=59)
                else:
                    end_time_dt = end_time

            # 处理语义记忆时间范围（仅对 semantic_memory 有效）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                if retrieve_mem_request.semantic_start_time:
                    semantic_start_dt = datetime.strptime(retrieve_mem_request.semantic_start_time, "%Y-%m-%d")
                if retrieve_mem_request.semantic_end_time:
                    semantic_end_dt = datetime.strptime(retrieve_mem_request.semantic_end_time, "%Y-%m-%d")
                if retrieve_mem_request.current_time:
                    current_time_dt = datetime.strptime(retrieve_mem_request.current_time, "%Y-%m-%d")

            # 调用 Milvus 的向量搜索（根据记忆类型传递不同的参数）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                # 语义记忆：支持时间范围和有效期过滤
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=semantic_start_dt,
                    end_time=semantic_end_dt,
                    current_time=current_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                )
            else:
                # 情景记忆和事件日志：使用 timestamp 过滤
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=start_time_dt,
                    end_time=end_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                )
            return search_results
        except Exception as e:
            logger.error(f"Error in get_vector_search_results: {e}")
            return {}

    # 混合记忆检索
    @trace_logger(operation_name="agentic_layer 混合记忆检索")
    async def retrieve_mem_hybrid(
        self, retrieve_mem_request: 'RetrieveMemRequest'
    ) -> RetrieveMemResponse:
        """基于关键词和向量的混合记忆检索

        Args:
            retrieve_mem_request: RetrieveMemRequest 包含检索参数

        Returns:
            RetrieveMemResponse 包含混合检索结果
        """
        try:
            logger.debug(
                f"retrieve_mem_hybrid called with retrieve_mem_request: {retrieve_mem_request}"
            )
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for retrieve_mem_hybrid"
                )

            query = retrieve_mem_request.query
            if not query:
                raise ValueError("query is required for retrieve_mem_hybrid")

            user_id = retrieve_mem_request.user_id
            top_k = retrieve_mem_request.top_k
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            logger.debug(
                f"retrieve_mem_hybrid called with query: {query}, user_id: {user_id}, top_k: {top_k}"
            )

            # 创建关键词检索请求
            keyword_request = RetrieveMemRequest(
                user_id=user_id,
                memory_types=retrieve_mem_request.memory_types,
                top_k=top_k,
                filters=retrieve_mem_request.filters,
                include_metadata=retrieve_mem_request.include_metadata,
                start_time=start_time,
                end_time=end_time,
                query=query,
            )

            # 创建向量检索请求
            vector_request = RetrieveMemRequest(
                user_id=user_id,
                memory_types=retrieve_mem_request.memory_types,
                top_k=top_k,
                filters=retrieve_mem_request.filters,
                include_metadata=retrieve_mem_request.include_metadata,
                start_time=start_time,
                end_time=end_time,
                query=query,
            )

            # 并行执行两种检索，获取原始搜索结果
            keyword_search_results = await self.get_keyword_search_results(
                keyword_request
            )
            vector_search_results = await self.get_vector_search_results(vector_request)

            logger.debug(f"关键词检索返回 {len(keyword_search_results)} 条原始结果")
            logger.debug(f"向量检索返回 {len(vector_search_results)} 条原始结果")

            # 合并原始搜索结果并进行rerank
            hybrid_result = await self._merge_and_rerank_search_results(
                keyword_search_results, vector_search_results, top_k, user_id, query
            )

            logger.debug(f"混合检索最终返回 {len(hybrid_result.memories)} 个群组")

            return hybrid_result

        except Exception as e:
            logger.error(f"Error in retrieve_mem_hybrid: {e}")
            return RetrieveMemResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_hybrid_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_hybrid",
                ),
                metadata=Metadata(
                    source="retrieve_mem_hybrid_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_hybrid",
                ),
            )

    def _extract_score_from_hit(self, hit: Dict[str, Any]) -> float:
        """从hit中提取得分

        Args:
            hit: 搜索结果hit

        Returns:
            得分
        """
        if '_score' in hit:
            return hit['_score']
        elif 'score' in hit:
            return hit['score']
        return 1.0

    async def _merge_and_rerank_search_results(
        self,
        keyword_search_results: List[Dict[str, Any]],
        vector_search_results: List[Dict[str, Any]],
        top_k: int,
        user_id: str,
        query: str,
    ) -> RetrieveMemResponse:
        """合并关键词和向量检索的原始搜索结果，并进行重新排序

        Args:
            keyword_search_results: 关键词检索的原始搜索结果
            vector_search_results: 向量检索的原始搜索结果
            top_k: 返回的最大群组数量
            user_id: 用户ID
            query: 查询文本

        Returns:
            RetrieveMemResponse: 合并和重新排序后的结果
        """
        # 提取搜索结果
        keyword_hits = keyword_search_results
        vector_hits = vector_search_results

        logger.debug(f"关键词检索原始结果: {len(keyword_hits)} 条")
        logger.debug(f"向量检索原始结果: {len(vector_hits)} 条")

        # 合并所有搜索结果并标记来源
        all_hits = []

        # 添加关键词检索结果，标记来源
        for hit in keyword_hits:
            hit_copy = hit.copy()
            hit_copy['_search_source'] = 'keyword'
            all_hits.append(hit_copy)

        # 添加向量检索结果，标记来源
        for hit in vector_hits:
            hit_copy = hit.copy()
            hit_copy['_search_source'] = 'vector'
            all_hits.append(hit_copy)

        logger.debug(f"合并后总结果数: {len(all_hits)} 条")

        # 使用rerank服务进行重排序
        try:
            rerank_service = get_rerank_service()
            reranked_hits = await rerank_service._rerank_all_hits(
                query, all_hits, top_k
            )

            logger.debug(f"使用rerank服务后取top_k结果数: {len(reranked_hits)} 条")

        except Exception as e:
            logger.error(f"使用rerank服务失败，回退到简单排序: {e}")
            # 如果rerank失败，回退到简单的得分排序
            reranked_hits = sorted(
                all_hits, key=self._extract_score_from_hit, reverse=True
            )[:top_k]

        # 对rerank后的结果进行分组处理
        memories, scores, importance_scores, original_data, total_count = (
            await self.group_by_groupid_stratagy(reranked_hits, source_type="hybrid")
        )

        # 构建最终结果
        return RetrieveMemResponse(
            memories=memories,
            scores=scores,
            importance_scores=importance_scores,
            original_data=original_data,
            total_count=total_count,
            has_more=False,
            query_metadata=Metadata(
                source="hybrid_retrieval",
                user_id=user_id,
                memory_type="retrieve_hybrid",
            ),
            metadata=Metadata(
                source="hybrid_retrieval",
                user_id=user_id,
                memory_type="retrieve_hybrid",
            ),
        )

    async def group_by_groupid_stratagy(
        self, search_results: List[Dict[str, Any]], source_type: str = "milvus"
    ) -> tuple:
        """通用的搜索结果分组处理策略

        Args:
            search_results: 搜索结果列表
            source_type: 数据源类型，支持 "es" 或 "milvus"

        Returns:
            tuple: (memories, scores, importance_scores, original_data, total_count)
        """
        memories_by_group = (
            {}
        )  # {group_id: {'memories': [Memory], 'scores': [float], 'importance_evidence': dict}}
        original_data_by_group = {}

        for hit in search_results:
            # 根据数据源类型提取数据
            if source_type == "es":
                # ES 搜索结果格式
                source = hit.get('_source', {})
                score = hit.get('_score', 1.0)
                user_id = source.get('user_id', '')
                group_id = source.get('group_id', '')
                timestamp_raw = source.get('timestamp', '')
                episode = source.get('episode', '')
                memcell_event_id_list = source.get('memcell_event_id_list', [])
                subject = source.get('subject', '')
                summary = source.get('summary', '')
                participants = source.get('participants', [])
                hit_id = source.get('event_id', '')
                search_source = hit.get('_search_source', 'keyword')  # 默认为关键词检索
            elif source_type == "hybrid":
                # 混合检索结果格式，需要根据_search_source字段判断
                search_source = hit.get('_search_source', 'unknown')
                if search_source == 'keyword':
                    # 关键词检索结果格式
                    source = hit.get('_source', {})
                    score = hit.get('_score', 1.0)
                    user_id = source.get('user_id', '')
                    group_id = source.get('group_id', '')
                    timestamp_raw = source.get('timestamp', '')
                    episode = source.get('episode', '')
                    memcell_event_id_list = source.get('memcell_event_id_list', [])
                    subject = source.get('subject', '')
                    summary = source.get('summary', '')
                    participants = source.get('participants', [])
                    hit_id = source.get('event_id', '')
                else:
                    # 向量检索结果格式
                    hit_id = hit.get('id', '')
                    score = hit.get('score', 1.0)
                    user_id = hit.get('user_id', '')
                    group_id = hit.get('group_id', '')
                    timestamp_raw = hit.get('timestamp')
                    episode = hit.get('episode', '')
                    metadata = hit.get('metadata', {})
                    memcell_event_id_list = metadata.get('memcell_event_id_list', [])
                    subject = metadata.get('subject', '')
                    summary = metadata.get('summary', '')
                    participants = metadata.get('participants', [])
            else:
                # Milvus 搜索结果格式
                hit_id = hit.get('id', '')
                score = hit.get('score', 1.0)
                user_id = hit.get('user_id', '')
                group_id = hit.get('group_id', '')
                timestamp_raw = hit.get('timestamp')
                episode = hit.get('episode', '')
                metadata = hit.get('metadata', {})
                memcell_event_id_list = metadata.get('memcell_event_id_list', [])
                subject = metadata.get('subject', '')
                summary = metadata.get('summary', '')
                participants = metadata.get('participants', [])
                search_source = 'vector'  # 默认为向量检索

            # 处理时间戳
            if timestamp_raw:
                if isinstance(timestamp_raw, datetime):
                    timestamp = timestamp_raw.replace(tzinfo=None)
                elif isinstance(timestamp_raw, (int, float)):
                    try:
                        timestamp = datetime.fromtimestamp(timestamp_raw)
                    except Exception as e:
                        logger.warning(
                            f"timestamp为数字但转换失败: {timestamp_raw}, error: {e}"
                        )
                        timestamp = datetime.now().replace(tzinfo=None)
                elif isinstance(timestamp_raw, str):
                    try:
                        timestamp = from_iso_format(timestamp_raw).replace(tzinfo=None)
                    except Exception as e:
                        logger.warning(
                            f"timestamp格式转换失败: {timestamp_raw}, error: {e}"
                        )
                        timestamp = datetime.now().replace(tzinfo=None)
                else:
                    logger.warning(
                        f"未知类型的timestamp_raw: {type(timestamp_raw)}, 使用当前时间"
                    )
                    timestamp = datetime.now().replace(tzinfo=None)
            else:
                timestamp = datetime.now().replace(tzinfo=None)

            # 获取 memcell 数据
            memcells = []
            if memcell_event_id_list:
                memcell_repo = get_bean_by_type(MemCellRawRepository)
                for event_id in memcell_event_id_list:
                    memcell = await memcell_repo.get_by_event_id(event_id)
                    if memcell:
                        memcells.append(memcell)
                    else:
                        logger.warning(f"未找到 memcell: event_id={event_id}")
                        continue

            # 为每个 memcell 添加原始数据
            for memcell in memcells:
                if group_id not in original_data_by_group:
                    original_data_by_group[group_id] = []
                original_data_by_group[group_id].append(memcell.original_data)

            # 创建 Memory 对象
            memory = Memory(
                memory_type="episode_summary",  # 情景记忆类型
                user_id=user_id,
                timestamp=timestamp,
                ori_event_id_list=[hit_id],
                subject=subject,
                summary=summary,
                episode=episode,
                group_id=group_id,
                participants=participants,
                memcell_event_id_list=memcell_event_id_list,
            )

            # 添加搜索来源信息到 extend 字段
            if not hasattr(memory, 'extend') or memory.extend is None:
                memory.extend = {}
            memory.extend['_search_source'] = search_source

            # 从 user_profiles 中读取 group_importance_evidence
            group_importance_evidence = None
            if user_id and group_id:
                try:
                    user_profile = await UserProfile.find_one(
                        UserProfile.user_id == user_id,
                        UserProfile.group_id == group_id,
                        sort=[("version", -1)],
                    )

                    if user_profile:
                        group_importance_evidence = (
                            user_profile.profile_data.get("group_importance_evidence")
                            if isinstance(user_profile.profile_data, dict)
                            else None
                        )
                        if group_importance_evidence:
                            if not hasattr(memory, 'extend') or memory.extend is None:
                                memory.extend = {}
                            memory.extend['group_importance_evidence'] = (
                                group_importance_evidence
                            )
                            logger.debug(
                                "为memory添加group_importance_evidence: user_id=%s, group_id=%s",
                                user_id,
                                group_id,
                            )
                except Exception as e:
                    logger.warning(
                        "读取 user_profiles 失败: user_id=%s, group_id=%s, error=%s",
                        user_id,
                        group_id,
                        e,
                    )

            # 按group_id分组
            if group_id not in memories_by_group:
                memories_by_group[group_id] = {
                    'memories': [],
                    'scores': [],
                    'importance_evidence': group_importance_evidence,
                }

            memories_by_group[group_id]['memories'].append(memory)
            memories_by_group[group_id]['scores'].append(score)  # 保存原始得分
            # 更新group_importance_evidence（如果当前memory有更新的证据）
            if group_importance_evidence:
                memories_by_group[group_id][
                    'importance_evidence'
                ] = group_importance_evidence

        def calculate_importance_score(importance_evidence):
            """计算群组重要性得分"""
            if not importance_evidence or not isinstance(importance_evidence, dict):
                return 0.0

            evidence_list = importance_evidence.get('evidence_list', [])
            if not evidence_list:
                return 0.0

            total_speak_count = 0
            total_refer_count = 0
            total_conversation_count = 0

            for evidence in evidence_list:
                if isinstance(evidence, dict):
                    total_speak_count += evidence.get('speak_count', 0)
                    total_refer_count += evidence.get('refer_count', 0)
                    total_conversation_count += evidence.get('conversation_count', 0)

            if total_conversation_count == 0:
                return 0.0

            return (total_speak_count + total_refer_count) / total_conversation_count

        # 为每个group内的memories按时间戳排序，并计算重要性得分
        group_scores = []
        for group_id, group_data in memories_by_group.items():
            # 按时间戳排序memories
            group_data['memories'].sort(
                key=lambda m: m.timestamp if m.timestamp else ''
            )

            # 计算重要性得分
            importance_score = calculate_importance_score(
                group_data['importance_evidence']
            )
            group_scores.append((group_id, importance_score))

        # 按重要性得分排序groups
        group_scores.sort(key=lambda x: x[1], reverse=True)

        # 构建最终结果
        memories = []
        scores = []
        importance_scores = []
        original_data = []
        for group_id, importance_score in group_scores:
            group_data = memories_by_group[group_id]
            group_memories = group_data['memories']
            group_scores_list = group_data['scores']
            group_original_data = original_data_by_group.get(group_id, [])
            memories.append({group_id: group_memories})
            # scores结构与memories保持一致：List[Dict[str, List[float]]]
            scores.append({group_id: group_scores_list})
            # original_data结构与memories保持一致：List[Dict[str, List[Dict[str, Any]]]]
            original_data.append({group_id: group_original_data})
            importance_scores.append(importance_score)

        total_count = sum(
            len(group_data['memories']) for group_data in memories_by_group.values()
        )
        return memories, scores, importance_scores, original_data, total_count
    
    # --------- Lightweight 检索（Embedding + BM25 + RRF）---------
    @trace_logger(operation_name="agentic_layer 轻量级检索")
    async def retrieve_lightweight(
        self,
        query: str,
        user_id: str = None,
        group_id: str = None,
        time_range_days: int = 365,
        top_k: int = 20,
        retrieval_mode: str = "rrf",  # "embedding" | "bm25" | "rrf"
        data_source: str = "episode",  # "episode" | "event_log" | "semantic_memory" | "profile"
        memory_scope: str = "all",  # "all" | "personal" | "group"
        current_time: Optional[datetime] = None,  # 当前时间，用于过滤有效期内的语义记忆
    ) -> Dict[str, Any]:
        """
        轻量级记忆检索（统一使用 Milvus/ES 检索）
        
        Args:
            query: 用户查询
            user_id: 用户ID（用于过滤）
            group_id: 群组ID（用于过滤；profile 数据源为必需）
            time_range_days: 时间范围天数
            top_k: 返回结果数量
            retrieval_mode: 检索模式
                - "embedding": 纯向量检索（通过 Milvus）
                - "bm25": 纯关键词检索（通过 ES）
                - "rrf": RRF 融合（默认，Milvus + ES）
            data_source: 数据源
                - "episode": 从 episode 检索（默认）
                - "event_log": 从 event_log 检索
                - "semantic_memory": 从语义记忆检索
                - "profile": 直接按 user_id + group_id 检索档案
            memory_scope: 记忆范围
                - "all": 所有记忆（默认，个人 + 群组）
                - "personal": 仅个人记忆（group_id 为空）
                - "group": 仅群组记忆（group_id 不为空）
            current_time: 当前时间，用于过滤有效期内的语义记忆（仅 data_source=semantic_memory 时有效）
            
        Returns:
            Dict 包含 memories, metadata
        """
        start_time = time.time()

        # 兼容旧参数名称
        if data_source == "memcell":
            data_source = "episode"

        if data_source == "profile":
            if not user_id or not group_id:
                raise ValueError("检索 profile 时必须同时提供 user_id 和 group_id")
            return await self._retrieve_profile_memories(
                user_id=user_id,
                group_id=group_id,
                top_k=top_k,
                start_time=start_time,
            )
        
        # 处理 memory_scope 参数逻辑
        # - "personal": 只传递 user_id 参数（不传 group_id）
        # - "group": 只传递 group_id 参数，同时设置 user_id="" 来过滤群组记忆
        # - "all": user_id 和 group_id 都传递
        scope_user_id = user_id
        scope_group_id = group_id
        
        if memory_scope == "personal":
            # 个人记忆：只传递 user_id，不传 group_id
            scope_group_id = None
        elif memory_scope == "group":
            # 群组记忆：episode 仍需传递 user_id=""，其余数据源只看 group_id
            if data_source == "episode":
                scope_user_id = ""  # 空字符串表示群组 episode
            else:
                scope_user_id = None  # 事件日志/语义记忆等只根据 group_id 过滤
        # "all": 保持原样，都传递
        
        return await self._retrieve_from_vector_stores(
            query=query,
            user_id=scope_user_id,
            group_id=scope_group_id,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            data_source=data_source,
            start_time=start_time,
            memory_scope=memory_scope,
            current_time=current_time,
        )
    
    async def _retrieve_from_vector_stores(
        self,
        query: str,
        user_id: str = None,
        group_id: str = None,
        top_k: int = 20,
        retrieval_mode: str = "rrf",
        data_source: str = "memcell",
        start_time: float = None,
        memory_scope: str = "all",
        current_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        统一的向量存储检索方法（支持 embedding、bm25、rrf 三种模式）
        
        Args:
            query: 查询文本
            user_id: 用户ID过滤
            group_id: 群组ID过滤
            top_k: 返回结果数量
            retrieval_mode: 检索模式（embedding/bm25/rrf）
            data_source: 数据源（memcell/event_log/semantic_memory）
            start_time: 开始时间（用于计算耗时）
            memory_scope: 记忆范围（all/personal/group）
            current_time: 当前时间，用于过滤有效期内的语义记忆（仅 data_source=semantic_memory 时有效）
            
        Returns:
            Dict 包含 memories, metadata
        """
        if start_time is None:
            start_time = time.time()
        
        try:
            # 1. Embedding 检索（通过 Milvus，根据 data_source 选择不同的 Repository）
            embedding_results = []
            embedding_count = 0
            
            if retrieval_mode in ["embedding", "rrf"]:
                # 根据 data_source 选择对应的 Milvus Repository
                if data_source == "semantic_memory":
                    from infra_layer.adapters.out.search.repository.semantic_memory_milvus_repository import (
                        SemanticMemoryMilvusRepository,
                    )
                    milvus_repo = get_bean_by_type(SemanticMemoryMilvusRepository)
                elif data_source == "event_log":
                    from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
                        EventLogMilvusRepository,
                    )
                    milvus_repo = get_bean_by_type(EventLogMilvusRepository)
                else:  # "episode"
                    milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
                
                vectorize_service = get_vectorize_service()
                
                # 生成查询向量
                query_vec = await vectorize_service.get_embedding(query)
                
                # 向量检索
                # 注意：为了确保能检索到足够的候选，增加 limit
                retrieval_limit = max(top_k * 10, 100)  # 至少 100 条候选
                
                # 根据 data_source 调用不同的 vector_search 参数
                if data_source == "semantic_memory":
                    # 语义记忆：支持 current_time 过滤有效期内的记忆
                    milvus_results = await milvus_repo.vector_search(
                        query_vector=query_vec,
                        user_id=user_id,
                        group_id=group_id,
                        current_time=current_time,
                        limit=retrieval_limit,
                    )
                else:
                    # 情景记忆和事件日志
                    milvus_results = await milvus_repo.vector_search(
                        query_vector=query_vec,
                        user_id=user_id,
                        group_id=group_id,
                        limit=retrieval_limit,
                    )
                
                # 处理 Milvus 检索结果（不再基于 user_id 是否为空进行二次过滤）
                for result in milvus_results:
                    l2_distance = result.get('score', 0)
                    cosine_sim = 1 - (l2_distance ** 2 / 2)
                    embedding_results.append((result, cosine_sim))
                
                # 按相似度排序
                embedding_results.sort(key=lambda x: x[1], reverse=True)
                logger.debug(f"Milvus 检索完成: data_source={data_source}, 结果数={len(embedding_results)}")
                embedding_count = len(embedding_results)
            
            # 2. BM25 检索（通过 Elasticsearch）
            bm25_results = []
            bm25_count = 0
            
            if retrieval_mode in ["bm25", "rrf"]:
                # 根据 data_source 选择对应的 ES Repository
                if data_source == "semantic_memory":
                    from infra_layer.adapters.out.search.repository.semantic_memory_es_repository import (
                        SemanticMemoryEsRepository,
                    )
                    es_repo = get_bean_by_type(SemanticMemoryEsRepository)
                elif data_source == "event_log":
                    from infra_layer.adapters.out.search.repository.event_log_es_repository import (
                        EventLogEsRepository,
                    )
                    es_repo = get_bean_by_type(EventLogEsRepository)
                else:  # "episode"
                    es_repo = get_bean_by_type(EpisodicMemoryEsRepository)
                
                # 使用 jieba 分词
                import jieba
                query_words = list(jieba.cut(query))
                
                # 调用 ES 检索
                # 注意：为了确保能检索到足够的候选，增加 size
                retrieval_size = max(top_k * 10, 100)  # 至少 100 条候选
                hits = await es_repo.multi_search(
                    query=query_words,
                    user_id=user_id,
                    group_id=group_id,
                    size=retrieval_size,
                )
                
                # 处理 ES 检索结果（不再基于 user_id 是否为空进行二次过滤）
                for hit in hits:
                    source = hit.get('_source', {})
                    bm25_score = hit.get('_score', 0)
                    result = {
                        'score': bm25_score,
                        'event_id': source.get('event_id', ''),
                        'user_id': source.get('user_id', ''),
                        'group_id': source.get('group_id', ''),
                        'timestamp': source.get('timestamp', ''),
                        'episode': source.get('episode', ''),
                        'search_content': source.get('search_content', []),
                        'metadata': source.get('extend', {}),
                    }
                    bm25_results.append((result, bm25_score))
                
                logger.debug(f"ES 检索完成: data_source={data_source}, 结果数={len(bm25_results)}")
                bm25_count = len(bm25_results)
            
            # 3. 根据模式返回结果
            if retrieval_mode == "embedding":
                # 纯向量检索
                final_results = embedding_results[:top_k]
                memories = [
                    {
                        'score': score,
                        'event_id': result.get('id', ''),
                        'user_id': result.get('user_id', ''),
                        'group_id': result.get('group_id', ''),
                        'timestamp': result.get('timestamp', ''),
                        'subject': result.get('metadata', {}).get('title', ''),
                        'episode': result.get('episode', '') if data_source == "episode" else result.get('content', '') if data_source == "semantic_memory" else result.get('atomic_fact', ''),
                        'summary': result.get('metadata', {}).get('summary', ''),
                        'evidence': result.get('evidence', '') if data_source == "semantic_memory" else '',
                        'metadata': result.get('metadata', {}),
                    }
                    for result, score in final_results
                ]
                
                metadata = {
                    "retrieval_mode": "embedding",
                    "data_source": data_source,
                    "embedding_candidates": embedding_count,
                    "final_count": len(memories),
                    "total_latency_ms": (time.time() - start_time) * 1000
                }
                
            elif retrieval_mode == "bm25":
                # 纯 BM25 检索
                final_results = bm25_results[:top_k]
                memories = [result for result, score in final_results]
                
                metadata = {
                    "retrieval_mode": "bm25",
                    "data_source": data_source,
                    "bm25_candidates": bm25_count,
                    "final_count": len(memories),
                    "total_latency_ms": (time.time() - start_time) * 1000
                }
                
            else:  # rrf
                # RRF 融合
                from agentic_layer.retrieval_utils import reciprocal_rank_fusion
                
                fused_results = reciprocal_rank_fusion(
                    embedding_results,
                    bm25_results,
                    k=60
                )
                
                final_results = fused_results[:top_k]
                
                # 统一格式
                memories = []
                for doc, rrf_score in final_results:
                    # doc 可能来自 Milvus 或 ES，需要统一格式
                    # 区分方法：Milvus 有 'id' 字段，ES 有 'event_id' 字段
                    if 'event_id' in doc and 'id' not in doc:
                        # 来自 ES 的结果（已经是标准格式）
                        memory = {
                            'score': rrf_score,
                            'event_id': doc.get('event_id', ''),
                            'user_id': doc.get('user_id', ''),
                            'group_id': doc.get('group_id', ''),
                            'timestamp': doc.get('timestamp', ''),
                            'subject': '',
                            'episode': doc.get('episode', ''),
                            'summary': '',
                            'evidence': doc.get('evidence', ''),
                            'metadata': doc.get('metadata', {}),
                        }
                    else:
                        # 来自 Milvus 的结果（需要转换字段名）
                        # 根据 data_source 获取正确的内容字段
                        content_field = 'episode'  # 默认
                        evidence_field = ''
                        if data_source == "semantic_memory":
                            content_field = 'content'
                            evidence_field = doc.get('evidence', '')
                        elif data_source == "event_log":
                            content_field = 'atomic_fact'
                        
                        memory = {
                            'score': rrf_score,
                            'event_id': doc.get('id', ''),  # Milvus 用 'id'
                            'user_id': doc.get('user_id', ''),
                            'group_id': doc.get('group_id', ''),
                            'timestamp': doc.get('timestamp', ''),
                            'subject': doc.get('metadata', {}).get('title', '') if isinstance(doc.get('metadata'), dict) else '',
                            'episode': doc.get(content_field, ''),
                            'summary': doc.get('metadata', {}).get('summary', '') if isinstance(doc.get('metadata'), dict) else '',
                            'evidence': evidence_field,
                            'metadata': doc.get('metadata', {}) if isinstance(doc.get('metadata'), dict) else {},
                        }
                    memories.append(memory)
                
                metadata = {
                    "retrieval_mode": "rrf",
                    "data_source": data_source,
                    "embedding_candidates": embedding_count,
                    "bm25_candidates": bm25_count,
                    "final_count": len(memories),
                    "total_latency_ms": (time.time() - start_time) * 1000
                }
            
            return {
                "memories": memories,
                "count": len(memories),
                "metadata": metadata,
            }
        
        except Exception as e:
            logger.error(f"向量存储检索失败: {e}", exc_info=True)
            return {
                "memories": [],
                "count": 0,
                "metadata": {
                    "retrieval_mode": retrieval_mode,
                    "data_source": data_source,
                    "error": str(e),
                    "total_latency_ms": (time.time() - start_time) * 1000
                }
            }

    async def _retrieve_profile_memories(
        self,
        user_id: str,
        group_id: str,
        top_k: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """从 user_profiles 集合直接读取用户画像"""
        doc = await UserProfile.find_one(
            UserProfile.user_id == user_id,
            UserProfile.group_id == group_id,
            sort=[("version", -1)],
        )

        memories: List[Dict[str, Any]] = []
        if doc:
            memories.append(
                {
                    "user_id": doc.user_id,
                    "group_id": doc.group_id,
                    "profile": doc.profile_data,
                    "scenario": doc.scenario,
                    "confidence": doc.confidence,
                    "version": doc.version,
                    "cluster_ids": doc.cluster_ids,
                    "memcell_count": doc.memcell_count,
                    "last_updated_cluster": doc.last_updated_cluster,
                    "updated_at": doc.updated_at.isoformat()
                    if doc.updated_at
                    else None,
                }
            )

        metadata = {
            "retrieval_mode": "direct",
            "data_source": "profile",
            "profile_count": len(memories),
            "total_latency_ms": (time.time() - start_time) * 1000,
        }

        return {
            "memories": memories[:top_k],
            "count": len(memories[:top_k]),
            "metadata": metadata,
        }
    
    # --------- Agentic 检索（LLM 引导的多轮检索）---------
    @trace_logger(operation_name="agentic_layer Agentic检索")
    async def retrieve_agentic(
        self,
        query: str,
        user_id: str = None,
        group_id: str = None,
        time_range_days: int = 365,
        top_k: int = 20,
        llm_provider = None,
        agentic_config = None,
    ) -> Dict[str, Any]:
        """Agentic 检索：LLM 引导的多轮智能检索
        
        流程：Round 1 (RRF检索) → Rerank → LLM判断 → Round 2 (多查询) → 融合 → Rerank
        """
        # 验证参数
        if llm_provider is None:
            raise ValueError("llm_provider is required for agentic retrieval")
        
        # 导入依赖
        from .agentic_utils import AgenticConfig, check_sufficiency, generate_multi_queries, format_documents_for_llm
        from .rerank_service import get_rerank_service
        
        # 使用默认配置
        if agentic_config is None:
            agentic_config = AgenticConfig()
        config = agentic_config
        
        start_time = time.time()
        metadata = {
            "retrieval_mode": "agentic",
            "is_multi_round": False,
            "round1_count": 0,
            "round1_reranked_count": 0,
            "is_sufficient": None,
            "reasoning": None,
            "missing_info": None,
            "refined_queries": None,
            "round2_count": 0,
            "final_count": 0,
            "total_latency_ms": 0.0,
        }
        
        logger.info(f"{'='*60}")
        logger.info(f"Agentic Retrieval: {query[:60]}...")
        logger.info(f"{'='*60}")
        
        try:
            # ========== Round 1: RRF 混合检索 ==========
            logger.info("Round 1: RRF retrieval...")
            
            round1_result = await self.retrieve_lightweight(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=config.round1_top_n,  # 20
                retrieval_mode="rrf",
                data_source="episode",
            )
            
            round1_memories = round1_result.get("memories", [])
            metadata["round1_count"] = len(round1_memories)
            metadata["round1_latency_ms"] = round1_result.get("metadata", {}).get("total_latency_ms", 0)
            
            logger.info(f"Round 1: Retrieved {len(round1_memories)} memories")
            
            if not round1_memories:
                logger.warning("Round 1 returned no results")
                metadata["total_latency_ms"] = (time.time() - start_time) * 1000
                return {"memories": [], "count": 0, "metadata": metadata}
            
            # ========== Rerank Round 1 结果 → Top 5 ==========
            if config.use_reranker:
                logger.info("Reranking Top 20 to Top 5 for sufficiency check...")
                rerank_service = get_rerank_service()
                
                # 转换格式用于 rerank
                candidates_for_rerank = [
                    {
                        "index": i,
                        "episode": mem.get("episode", ""),
                        "summary": mem.get("summary", ""),
                        "subject": mem.get("subject", ""),
                        "score": mem.get("score", 0),
                    }
                    for i, mem in enumerate(round1_memories)
                ]
                
                reranked_hits = await rerank_service._rerank_all_hits(
                    query, candidates_for_rerank, top_k=config.round1_rerank_top_n
                )
                
                # 提取 Top 5 用于 LLM 判断
                top5_for_llm = []
                for hit in reranked_hits[:config.round1_rerank_top_n]:
                    idx = hit.get("index", 0)
                    if 0 <= idx < len(round1_memories):
                        mem = round1_memories[idx]
                        # 转换为 (candidate, score) 格式供 LLM 使用
                        top5_for_llm.append((mem, hit.get("relevance_score", 0)))
                
                metadata["round1_reranked_count"] = len(top5_for_llm)
                logger.info(f"Rerank: Got Top {len(top5_for_llm)} for sufficiency check")
            else:
                # 不使用 reranker，直接取前 5
                top5_for_llm = [(mem, mem.get("score", 0)) for mem in round1_memories[:config.round1_rerank_top_n]]
                metadata["round1_reranked_count"] = len(top5_for_llm)
                logger.info("No Rerank: Using original Top 5")
            
            if not top5_for_llm:
                logger.warning("No results for sufficiency check")
                metadata["total_latency_ms"] = (time.time() - start_time) * 1000
                return round1_result
            
            # ========== LLM 判断充分性 ==========
            logger.info("LLM: Checking sufficiency on Top 5...")
            
            is_sufficient, reasoning, missing_info = await check_sufficiency(
                query=query,
                results=top5_for_llm,
                llm_provider=llm_provider,
                max_docs=config.round1_rerank_top_n
            )
            
            metadata["is_sufficient"] = is_sufficient
            metadata["reasoning"] = reasoning
            metadata["missing_info"] = missing_info
            
            logger.info(f"LLM Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}")
            logger.info(f"LLM Reasoning: {reasoning}")
            
            # ========== 如果充分：直接返回 Round 1 结果 ==========
            if is_sufficient:
                logger.info("Decision: Sufficient! Using Round 1 results")
                metadata["final_count"] = len(round1_memories)
                metadata["total_latency_ms"] = (time.time() - start_time) * 1000
                
                round1_result["metadata"] = metadata
                logger.info(f"Complete: Latency {metadata['total_latency_ms']:.0f}ms")
                return round1_result
            
            # ========== Round 2: LLM 生成多个改进查询 ==========
            metadata["is_multi_round"] = True
            logger.info("Decision: Insufficient, entering Round 2")
            
            if missing_info:
                logger.info(f"Missing: {', '.join(missing_info)}")
            
            if config.enable_multi_query:
                logger.info("LLM: Generating multiple refined queries...")
                
                refined_queries, query_strategy = await generate_multi_queries(
                    original_query=query,
                    results=top5_for_llm,
                    missing_info=missing_info,
                    llm_provider=llm_provider,
                    max_docs=config.round1_rerank_top_n,
                    num_queries=config.num_queries
                )
                
                metadata["refined_queries"] = refined_queries
                metadata["query_strategy"] = query_strategy
                metadata["num_queries"] = len(refined_queries)
                
                logger.info(f"Generated {len(refined_queries)} queries")
                for i, q in enumerate(refined_queries, 1):
                    logger.debug(f"  Query {i}: {q[:80]}...")
            else:
                # 单查询模式
                refined_queries = [query]
                metadata["refined_queries"] = refined_queries
                metadata["num_queries"] = 1
            
            # ========== Round 2: 并行执行多查询检索 ==========
            logger.info(f"Round 2: Executing {len(refined_queries)} queries in parallel...")
            
            # 并行调用 retrieve_lightweight
            round2_tasks = [
                self.retrieve_lightweight(
                    query=q,
                    user_id=user_id,
                    group_id=group_id,
                    time_range_days=time_range_days,
                    top_k=config.round2_per_query_top_n,  # 每个查询 50 条
                    retrieval_mode="rrf",
                    data_source="episode",
                )
                for q in refined_queries
            ]
            
            round2_results_list = await asyncio.gather(*round2_tasks, return_exceptions=True)
            
            # 收集所有查询的结果
            all_round2_memories = []
            for i, result in enumerate(round2_results_list, 1):
                if isinstance(result, Exception):
                    logger.error(f"Query {i} failed: {result}")
                    continue
                
                memories = result.get("memories", [])
                if memories:
                    all_round2_memories.extend(memories)
                    logger.debug(f"Query {i}: Retrieved {len(memories)} memories")
            
            logger.info(f"Round 2: Total retrieved {len(all_round2_memories)} memories before dedup")
            
            # ========== 去重和融合 ==========
            logger.info("Merge: Deduplicating and combining Round 1 + Round 2...")
            
            # 去重：使用 event_id
            round1_event_ids = {mem.get("event_id") for mem in round1_memories}
            round2_unique = [
                mem for mem in all_round2_memories
                if mem.get("event_id") not in round1_event_ids
            ]
            
            # 合并：Round 1 (20) + Round 2 去重后的结果（最多取到总数 40）
            combined_memories = round1_memories.copy()
            needed_from_round2 = config.combined_total - len(combined_memories)
            combined_memories.extend(round2_unique[:needed_from_round2])
            
            metadata["round2_count"] = len(round2_unique[:needed_from_round2])
            logger.info(f"Merge: Round1={len(round1_memories)}, Round2_unique={len(round2_unique[:needed_from_round2])}, Total={len(combined_memories)}")
            
            # ========== Final Rerank ==========
            if config.use_reranker and len(combined_memories) > 0:
                logger.info(f"Rerank: Reranking {len(combined_memories)} memories...")
                
                rerank_service = get_rerank_service()
                
                # 转换格式
                candidates_for_rerank = [
                    {
                        "index": i,
                        "episode": mem.get("episode", ""),
                        "summary": mem.get("summary", ""),
                        "subject": mem.get("subject", ""),
                        "score": mem.get("score", 0),
                    }
                    for i, mem in enumerate(combined_memories)
                ]
                
                reranked_hits = await rerank_service._rerank_all_hits(
                    query,  # 使用原始查询
                    candidates_for_rerank,
                    top_k=config.final_top_n
                )
                
                # 提取最终 Top 20
                final_memories = []
                for hit in reranked_hits[:config.final_top_n]:
                    idx = hit.get("index", 0)
                    if 0 <= idx < len(combined_memories):
                        mem = combined_memories[idx].copy()
                        mem["score"] = hit.get("relevance_score", mem.get("score", 0))
                        final_memories.append(mem)
                
                logger.info(f"Rerank: Final Top {len(final_memories)} selected")
            else:
                # 不使用 Reranker，直接返回 Top N
                final_memories = combined_memories[:config.final_top_n]
                logger.info(f"No Rerank: Returning Top {len(final_memories)}")
            
            metadata["final_count"] = len(final_memories)
            metadata["total_latency_ms"] = (time.time() - start_time) * 1000
            
            logger.info(f"Complete: Final {len(final_memories)} memories | Latency {metadata['total_latency_ms']:.0f}ms")
            logger.info(f"{'='*60}\n")
            
            return {
                "memories": final_memories,
                "count": len(final_memories),
                "metadata": metadata,
            }
        
        except Exception as e:
            logger.error(f"Agentic retrieval failed: {e}", exc_info=True)
            
            # 降级到 lightweight
            logger.warning("Falling back to lightweight retrieval")
            
            fallback_result = await self.retrieve_lightweight(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=top_k,
                retrieval_mode="rrf",
                data_source="episode",
            )
            
            fallback_result["metadata"]["retrieval_mode"] = "agentic_fallback"
            fallback_result["metadata"]["fallback_reason"] = str(e)
            
            return fallback_result
