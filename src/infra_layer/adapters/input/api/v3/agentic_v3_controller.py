"""
Agentic Layer V3 控制器

提供专门用于处理群聊记忆的 RESTful API 路由
直接接收简单直接的消息格式，逐条处理并存储
"""

import logging
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from agentic_layer.schemas import RetrieveMethod
from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import BaseController, post
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.memory_manager import MemoryManager
from agentic_layer.converter import (
    _handle_conversation_format,
    convert_dict_to_fetch_mem_request,
    convert_dict_to_retrieve_mem_request,
)
from agentic_layer.dtos.memory_query import ConversationMetaRequest, UserDetail
from infra_layer.adapters.input.api.mapper.group_chat_converter import (
    convert_simple_message_to_memorize_input,
)
from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
    ConversationMeta,
    UserDetailModel,
)
from infra_layer.adapters.out.persistence.repository.conversation_meta_raw_repository import (
    ConversationMetaRawRepository,
)
from component.redis_provider import RedisProvider
import json

logger = logging.getLogger(__name__)


@controller("agentic_v3_controller", primary=True)
class AgenticV3Controller(BaseController):
    """
    Agentic Layer V3 API 控制器

    提供专门用于群聊记忆的接口：
    - memorize: 存储单条消息记忆
    - retrieve_lightweight: 轻量级检索（Embedding + BM25 + RRF）
    """

    def __init__(self, conversation_meta_repository: ConversationMetaRawRepository):
        """初始化控制器"""
        super().__init__(
            prefix="/api/v3/agentic",
            tags=["Agentic Layer V3"],
            default_auth="none",  # 根据实际需求调整认证策略
        )
        self.memory_manager = MemoryManager()
        self.conversation_meta_repository = conversation_meta_repository
        # 获取 RedisProvider
        self.redis_provider = get_bean_by_type(RedisProvider)
        logger.info("AgenticV3Controller initialized with MemoryManager and ConversationMetaRepository")

    @post(
        "/memorize",
        response_model=Dict[str, Any],
        summary="存储单条群聊消息记忆",
        description="""
        接收简单直接的单条消息格式并存储为记忆
        
        ## 功能说明：
        - 接收简单直接的单条消息数据（无需预转换）
        - 将单条消息提取为记忆单元（memcells）
        - 适用于实时消息处理场景
        - 返回已保存的记忆列表
        
        ## 输入格式（简单直接）：
        ```json
        {
          "group_id": "group_123",
          "group_name": "项目讨论组",
          "message_id": "msg_001",
          "create_time": "2025-01-15T10:00:00+08:00",
          "sender": "user_001",
          "sender_name": "张三",
          "content": "今天讨论下新功能的技术方案",
          "refer_list": ["msg_000"]
        }
        ```
        
        ## 字段说明：
        - **group_id** (可选): 群组ID
        - **group_name** (可选): 群组名称
        - **message_id** (必需): 消息ID
        - **create_time** (必需): 消息创建时间（ISO 8601格式）
        - **sender** (必需): 发送者用户ID
        - **sender_name** (可选): 发送者名称
        - **content** (必需): 消息内容
        - **refer_list** (可选): 引用的消息ID列表
        
        ## 与其他接口的区别：
        - **V3 /memorize**: 简单直接的单条消息格式（本接口，推荐）
        - **V2 /memorize**: 接收内部格式，需要外部转换
        
        ## 使用场景：
        - 实时消息流处理
        - 聊天机器人集成
        - 消息队列消费
        - 单条消息导入
        """,
        responses={
            200: {
                "description": "成功存储记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "记忆存储成功，共保存 1 条记忆",
                            "result": {
                                "saved_memories": [
                                    {
                                        "memory_type": "episode_summary",
                                        "user_id": "user_001",
                                        "group_id": "group_123",
                                        "timestamp": "2025-01-15T10:00:00",
                                        "content": "用户讨论了新功能的技术方案",
                                    }
                                ],
                                "count": 1,
                            },
                        }
                    }
                },
            },
            400: {
                "description": "请求参数错误",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.INVALID_PARAMETER.value,
                            "message": "数据格式错误：缺少必需字段 message_id",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/memorize",
                        }
                    }
                },
            },
            500: {
                "description": "服务器内部错误",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.SYSTEM_ERROR.value,
                            "message": "存储记忆失败，请稍后重试",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/memorize",
                        }
                    }
                },
            },
        },
    )
    async def memorize_single_message(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        存储单条消息记忆数据

        接收简单直接的单条消息格式，通过 group_chat_converter 转换并存储

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 记忆存储响应，包含已保存的记忆列表

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 1. 从请求中获取 JSON body（简单直接的格式）
            message_data = await fastapi_request.json()
            logger.info("收到 V3 memorize 请求（单条消息）")

            # 2. 保存原始消息到 Redis（用于历史累积）
            group_id = message_data.get("group_id")
            if group_id:
                redis_key = f"chat_history:{group_id}"

                # 将消息保存到 Redis List（左侧推入，保持时间顺序）
                await self.redis_provider.lpush(redis_key, json.dumps(message_data))
                # 设置过期时间为 24 小时
                await self.redis_provider.expire(redis_key, 86400)
                logger.debug("消息已保存到 Redis: group_id=%s", group_id)
        

            # 3. 使用 group_chat_converter 转换为内部格式
            logger.info("开始转换简单消息格式到内部格式")
            memorize_input = convert_simple_message_to_memorize_input(message_data)

            # 提取元信息用于日志
            group_name = memorize_input.get("group_name")

            logger.info("转换完成: group_id=%s, group_name=%s", group_id, group_name)

            # 4. 转换为 MemorizeRequest 对象并调用 memory_manager
            logger.info("开始处理记忆请求")
            memorize_request = await _handle_conversation_format(memorize_input)
            memories = await self.memory_manager.memorize(memorize_request)

            # 5. 返回统一格式的响应
            memory_count = len(memories) if memories else 0
            logger.info("处理记忆请求完成，保存了 %s 条记忆", memory_count)

            return {
                "status": ErrorStatus.OK.value,
                "message": f"记忆存储成功，共保存 {memory_count} 条记忆",
                "result": {"saved_memories": memories, "count": memory_count},
            }

        except ValueError as e:
            logger.error("V3 memorize 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 memorize 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="存储记忆失败，请稍后重试"
            ) from e
    
    @post(
        "/retrieve_lightweight",
        response_model=Dict[str, Any],
        summary="轻量级记忆检索（Embedding + BM25 + RRF）",
        description="""
        轻量级记忆检索接口，使用 Embedding + BM25 + RRF 融合策略
        
        ## 功能说明：
        - 并行执行向量检索和关键词检索
        - 使用 RRF（Reciprocal Rank Fusion）融合结果
        - 速度快，适合实时场景
        
        ## 输入格式：
        ```json
        {
          "query": "北京旅游美食",
          "user_id": "default",
          "group_id": "assistant",
          "time_range_days": 365,
          "top_k": 20,
          "retrieval_mode": "rrf",
          "data_source": "memcell",
          "memory_scope": "all"
        }
        ```
        
        ## 字段说明：
        - **query** (必需): 用户查询
        - **user_id** (可选): 用户ID（用于过滤）
        - **group_id** (可选): 群组ID（用于过滤）
        - **time_range_days** (可选): 时间范围天数（默认365天）
        - **top_k** (可选): 返回结果数量（默认20）
        - **retrieval_mode** (可选): 检索模式
          * "rrf": RRF 融合（默认）
          * "embedding": 纯向量检索
          * "bm25": 纯关键词检索
        - **data_source** (可选): 数据源
          * "memcell": 从 MemCell.episode 检索（默认）
          * "event_log": 从 event_log.atomic_fact 检索
          * "semantic_memory": 从语义记忆检索
        - **memory_scope** (可选): 记忆范围
          * "all": 所有记忆（默认，包含个人和群组）
          * "personal": 仅个人记忆（personal_episode/personal_event_log/personal_semantic_memory）
          * "group": 仅群组记忆（episode/event_log/semantic_memory）
        - **radius** (可选): COSINE 相似度阈值，范围 [-1, 1]，默认 0.6
          * 只返回相似度 >= radius 的结果
          * 影响向量检索部分（embedding/rrf 模式）的结果质量
        
        ## 返回格式：
        ```json
        {
          "status": "ok",
          "message": "检索成功，找到 10 条记忆",
          "result": {
            "memories": [...],
            "count": 10,
            "metadata": {
              "retrieval_mode": "lightweight",
              "emb_count": 15,
              "bm25_count": 12,
              "final_count": 10,
              "total_latency_ms": 123.45
            }
          }
        }
        ```
        """,
    )
    async def retrieve_lightweight(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        轻量级记忆检索（Embedding + BM25 + RRF 融合）
        
        Args:
            fastapi_request: FastAPI 请求对象
            
        Returns:
            Dict[str, Any]: 检索结果响应
        """
        try:
            # 1. 解析请求参数
            request_data = await fastapi_request.json()
            query = request_data.get("query")
            user_id = request_data.get("user_id")
            group_id = request_data.get("group_id")
            time_range_days = request_data.get("time_range_days", 365)
            top_k = request_data.get("top_k", 20)
            retrieval_mode = request_data.get("retrieval_mode", "rrf")
            data_source = request_data.get("data_source", "memcell")
            memory_scope = request_data.get("memory_scope", "all")  # 新增参数
            radius = request_data.get("radius")  # COSINE 相似度阈值（可选）
            
            if not query:
                raise ValueError("缺少必需参数：query")
            
            logger.info(
                f"收到 lightweight 检索请求: query={query}, group_id={group_id}, "
                f"mode={retrieval_mode}, source={data_source}, scope={memory_scope}, top_k={top_k}"
            )
            
            # 2. 调用 memory_manager 的 lightweight 检索
            result = await self.memory_manager.retrieve_lightweight(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                data_source=data_source,
                memory_scope=memory_scope,
                radius=radius,
            )
            
            # 3. 返回统一格式
            return {
                "status": ErrorStatus.OK.value,
                "message": f"检索成功，找到 {result['count']} 条记忆",
                "result": result,
            }
        
        except ValueError as e:
            logger.error("V3 retrieve_lightweight 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("V3 retrieve_lightweight 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="检索失败，请稍后重试"
            ) from e
    
    @post(
        "/retrieve_agentic",
        response_model=Dict[str, Any],
        summary="Agentic 记忆检索（LLM 引导的多轮检索）",
        description="""
        Agentic 记忆检索接口，使用 LLM 引导的多轮智能检索
        
        ## 功能说明：
        - 使用 LLM 判断检索充分性
        - 自动进行多轮检索和查询改进
        - 使用 Rerank 提升结果质量
        - 适合需要深度理解的复杂查询
        
        ## 检索流程：
        1. Round 1: RRF 混合检索（Embedding + BM25）
        2. Rerank 优化结果
        3. LLM 判断是否充分
        4. 如果不充分：生成多个改进查询
        5. Round 2: 多查询并行检索
        6. 融合并 Rerank 返回最终结果
        
        ## 输入格式：
        ```json
        {
          "query": "用户喜欢吃什么？",
          "user_id": "default",
          "group_id": "assistant",
          "time_range_days": 365,
          "top_k": 20,
          "llm_config": {
            "api_key": "your_api_key",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
          }
        }
        ```
        
        ## 字段说明：
        - **query** (必需): 用户查询
        - **user_id** (可选): 用户ID（用于过滤）
        - **group_id** (可选): 群组ID（用于过滤）
        - **time_range_days** (可选): 时间范围天数（默认365天）
        - **top_k** (可选): 返回结果数量（默认20）
        - **llm_config** (可选): LLM 配置
          * api_key: LLM API Key（可选，默认使用环境变量）
          * base_url: LLM API 地址（可选，默认 OpenRouter）
          * model: LLM 模型（可选，默认 gpt-4o-mini）
        
        ## 返回格式：
        ```json
        {
          "status": "ok",
          "message": "Agentic 检索成功，找到 15 条记忆",
          "result": {
            "memories": [...],
            "count": 15,
            "metadata": {
              "retrieval_mode": "agentic",
              "is_multi_round": true,
              "round1_count": 20,
              "is_sufficient": false,
              "reasoning": "需要更多关于饮食偏好的具体信息",
              "refined_queries": ["用户最喜欢的菜系？", "用户不喜欢吃什么？"],
              "round2_count": 40,
              "final_count": 15,
              "total_latency_ms": 2345.67
            }
          }
        }
        ```
        
        ## 使用场景：
        - 复杂问题回答
        - 深度信息挖掘
        - 多维度记忆检索
        - 智能对话系统
        
        ## 注意事项：
        - 需要配置 LLM API Key
        - 检索耗时较长（通常 2-5 秒）
        - 会产生 LLM API 调用费用
        """,
    )
    async def retrieve_agentic(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        Agentic 记忆检索（LLM 引导的多轮智能检索）
        
        Args:
            fastapi_request: FastAPI 请求对象
            
        Returns:
            Dict[str, Any]: 检索结果响应
        """
        try:
            # 1. 解析请求参数
            request_data = await fastapi_request.json()
            query = request_data.get("query")
            user_id = request_data.get("user_id")
            group_id = request_data.get("group_id")
            time_range_days = request_data.get("time_range_days", 365)
            top_k = request_data.get("top_k", 20)
            llm_config = request_data.get("llm_config", {})
            
            if not query:
                raise ValueError("缺少必需参数：query")
            
            logger.info(
                f"收到 agentic 检索请求: query={query}, group_id={group_id}, top_k={top_k}"
            )
            
            # 2. 创建 LLM Provider
            from memory_layer.llm.llm_provider import LLMProvider
            import os
            
            # 从请求或环境变量获取配置
            api_key = llm_config.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = llm_config.get("base_url") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            model = llm_config.get("model") or os.getenv("LLM_MODEL", "qwen/qwen3-235b-a22b-2507")
            
            if not api_key:
                raise ValueError("缺少 LLM API Key，请在 llm_config.api_key 中提供或设置环境变量 OPENROUTER_API_KEY/OPENAI_API_KEY")
            
            # 创建 LLM Provider（使用 OpenAI 兼容接口）
            llm_provider = LLMProvider(
                provider_type="openai",
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.3,
                max_tokens=2048,
            )
            
            logger.info(f"使用 LLM: {model} @ {base_url}")
            
            # 3. 调用 memory_manager 的 agentic 检索
            result = await self.memory_manager.retrieve_agentic(
                query=query,
                user_id=user_id,
                group_id=group_id,
                time_range_days=time_range_days,
                top_k=top_k,
                llm_provider=llm_provider,
                agentic_config=None,  # 使用默认配置
            )
            
            # 4. 返回统一格式
            return {
                "status": ErrorStatus.OK.value,
                "message": f"Agentic 检索成功，找到 {result['count']} 条记忆",
                "result": result,
            }
        
        except ValueError as e:
            logger.error("V3 retrieve_agentic 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error("V3 retrieve_agentic 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Agentic 检索失败，请稍后重试"
            ) from e