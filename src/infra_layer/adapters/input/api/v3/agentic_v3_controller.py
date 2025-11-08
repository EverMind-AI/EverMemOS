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

logger = logging.getLogger(__name__)


@controller("agentic_v3_controller", primary=True)
class AgenticV3Controller(BaseController):
    """
    Agentic Layer V3 API 控制器

    提供完整的记忆管理接口：
    - memorize: 逐条接收简单直接的单条消息并存储为记忆
    - conversation-meta: 保存对话元数据
    - fetch: 使用 KV 方式获取用户核心记忆
    - retrieve: 支持关键词/向量/混合三种检索方法
    - retrieve_keyword: 基于关键词的 BM25 检索
    - retrieve_vector: 基于语义向量相似度检索
    - retrieve_hybrid: 结合关键词和向量的混合检索
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

            # 2. 使用 group_chat_converter 转换为内部格式
            logger.info("开始转换简单消息格式到内部格式")
            memorize_input = convert_simple_message_to_memorize_input(message_data)

            # 提取元信息用于日志
            group_id = memorize_input.get("group_id")
            group_name = memorize_input.get("group_name")

            logger.info("转换完成: group_id=%s, group_name=%s", group_id, group_name)

            # 3. 转换为 MemorizeRequest 对象并调用 memory_manager
            logger.info("开始处理记忆请求")
            memorize_request = await _handle_conversation_format(memorize_input)
            memories = await self.memory_manager.memorize(memorize_request)

            # 4. 返回统一格式的响应
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
        "/conversation-meta",
        response_model=Dict[str, Any],
        summary="保存对话元数据",
        description="""
        保存对话的元数据信息，包括场景、参与者、标签等
        """
    )
    async def save_conversation_meta(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        保存对话元数据

        接收 ConversationMetaRequest 格式的数据，转换为 ConversationMeta ODM 模型并保存到 MongoDB

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 保存响应，包含已保存的元数据信息

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 1. 从请求中获取 JSON body
            request_data = await fastapi_request.json()
            logger.info("收到 V3 conversation-meta 保存请求: group_id=%s", request_data.get("group_id"))

            # 2. 解析为 ConversationMetaRequest
            # 处理 user_details 的转换
            user_details_data = request_data.get("user_details", {})
            user_details = {}
            for user_id, detail_data in user_details_data.items():
                user_details[user_id] = UserDetail(
                    full_name=detail_data["full_name"],
                    role=detail_data["role"],
                    extra=detail_data.get("extra", {}),
                )

            conversation_meta_request = ConversationMetaRequest(
                version=request_data["version"],
                scene=request_data["scene"],
                scene_desc=request_data["scene_desc"],
                name=request_data["name"],
                description=request_data["description"],
                group_id=request_data["group_id"],
                created_at=request_data["created_at"],
                default_timezone=request_data["default_timezone"],
                user_details=user_details,
                tags=request_data.get("tags", []),
            )

            logger.info("解析 ConversationMetaRequest 成功: group_id=%s", conversation_meta_request.group_id)

            # 3. 转换为 ConversationMeta ODM 模型
            user_details_model = {}
            for user_id, detail in conversation_meta_request.user_details.items():
                user_details_model[user_id] = UserDetailModel(
                    full_name=detail.full_name,
                    role=detail.role,
                    extra=detail.extra,
                )

            conversation_meta = ConversationMeta(
                version=conversation_meta_request.version,
                scene=conversation_meta_request.scene,
                scene_desc=conversation_meta_request.scene_desc,
                name=conversation_meta_request.name,
                description=conversation_meta_request.description,
                group_id=conversation_meta_request.group_id,
                conversation_created_at=conversation_meta_request.created_at,
                default_timezone=conversation_meta_request.default_timezone,
                user_details=user_details_model,
                tags=conversation_meta_request.tags,
            )

            # 4. 使用 upsert 方式保存（如果 group_id 已存在则更新）
            logger.info("开始保存对话元数据到 MongoDB")
            saved_meta = await self.conversation_meta_repository.upsert_by_group_id(
                group_id=conversation_meta.group_id,
                conversation_data={
                    "version": conversation_meta.version,
                    "scene": conversation_meta.scene,
                    "scene_desc": conversation_meta.scene_desc,
                    "name": conversation_meta.name,
                    "description": conversation_meta.description,
                    "conversation_created_at": conversation_meta.conversation_created_at,
                    "default_timezone": conversation_meta.default_timezone,
                    "user_details": conversation_meta.user_details,
                    "tags": conversation_meta.tags,
                },
            )

            if not saved_meta:
                raise HTTPException(
                    status_code=500, detail="保存对话元数据失败"
                )

            logger.info("保存对话元数据成功: id=%s, group_id=%s", saved_meta.id, saved_meta.group_id)

            # 5. 返回成功响应
            return {
                "status": ErrorStatus.OK.value,
                "message": "对话元数据保存成功",
                "result": {
                    "id": str(saved_meta.id),
                    "group_id": saved_meta.group_id,
                    "scene": saved_meta.scene,
                    "name": saved_meta.name,
                    "version": saved_meta.version,
                    "created_at": saved_meta.created_at.isoformat() if saved_meta.created_at else None,
                    "updated_at": saved_meta.updated_at.isoformat() if saved_meta.updated_at else None,
                },
            }

        except KeyError as e:
            logger.error("V3 conversation-meta 请求缺少必需字段: %s", e)
            raise HTTPException(
                status_code=400, detail=f"缺少必需字段: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("V3 conversation-meta 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 conversation-meta 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="保存对话元数据失败，请稍后重试"
            ) from e

    @post(
        "/fetch",
        response_model=Dict[str, Any],
        summary="获取用户记忆",
        description="""
        通过 KV 方式获取用户的核心记忆数据
        
        ## 功能说明：
        - 根据用户ID直接获取存储的核心记忆
        - 支持多种记忆类型：基础记忆、用户画像、偏好设置等
        - 支持分页和排序
        - 适用于需要快速获取用户固定记忆集合的场景
        
        ## 记忆类型说明：
        - **base_memory**: 基础记忆，用户的基本信息和常用数据
        - **profile**: 用户画像，包含用户的特征和属性
        - **preference**: 用户偏好，包含用户的喜好和设置
        - **episode_summary**: 情景记忆摘要
        - **multiple**: 多类型（默认），包含 base_memory、profile、preference
        
        ## 使用场景：
        - 用户个人资料展示
        - 个性化推荐系统
        - 用户偏好设置加载
        """,
        responses={
            200: {
                "description": "成功获取记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "记忆获取成功",
                            "result": {
                                "memories": [
                                    {
                                        "memory_type": "base_memory",
                                        "user_id": "user_123",
                                        "timestamp": "2024-01-15T10:30:00",
                                        "content": "用户喜欢喝咖啡",
                                        "summary": "咖啡偏好",
                                    }
                                ],
                                "total_count": 100,
                                "has_more": False,
                                "metadata": {
                                    "source": "fetch_mem_service",
                                    "user_id": "user_123",
                                    "memory_type": "fetch",
                                },
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
                            "message": "user_id 不能为空",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/fetch",
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
                            "message": "获取记忆失败，请稍后重试",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/fetch",
                        }
                    }
                },
            },
        },
    )
    async def fetch_memories(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        获取用户记忆数据

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 记忆获取响应

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 从请求中获取 JSON body
            body = await fastapi_request.json()
            logger.info(
                "收到 V3 fetch 请求: user_id=%s, memory_type=%s",
                body.get("user_id"),
                body.get("memory_type"),
            )

            # 直接使用 converter 转换
            fetch_request = convert_dict_to_fetch_mem_request(body)

            # 调用 memory_manager 的 fetch_mem 方法
            response = await self.memory_manager.fetch_mem(fetch_request)

            # 返回统一格式的响应
            memory_count = len(response.memories) if response.memories else 0
            logger.info(
                "V3 fetch 请求处理完成: user_id=%s, 返回 %s 条记忆",
                body.get("user_id"),
                memory_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"记忆获取成功，共获取 {memory_count} 条记忆",
                "result": response,
            }

        except ValueError as e:
            logger.error("V3 fetch 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 fetch 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="获取记忆失败，请稍后重试"
            ) from e

    @post(
        "/retrieve",
        response_model=Dict[str, Any],
        summary="检索相关记忆（支持关键词/向量/混合检索）",
        description="""
        基于查询文本使用关键词、向量或混合方法检索相关的记忆数据
        
        ## 功能说明：
        - 根据查询文本查找最相关的记忆
        - 支持关键词（BM25）、向量相似度、混合检索三种方法
        - 支持时间范围过滤
        - 返回结果按群组组织，并包含相关性评分
        - 适用于需要精确匹配或语义检索的场景
        
        ## 检索方法说明：
        - **keyword**: 基于关键词的 BM25 检索，适合精确匹配，速度快（默认方法）
        - **vector**: 基于语义向量的相似度检索，适合模糊查询和语义相似查询
        - **hybrid**: 混合检索策略，结合关键词和向量检索的优势（推荐）
        
        ## 返回结果说明：
        - 记忆按群组（group）组织返回
        - 每个群组包含多条相关记忆，按时间排序
        - 群组按重要性得分排序，最重要的群组排在前面
        - 每条记忆都有相关性得分，表示与查询的匹配程度
        
        ## 使用场景：
        - 对话上下文理解
        - 智能问答系统
        - 相关内容推荐
        - 记忆线索追溯
        """,
        responses={
            200: {
                "description": "成功检索记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "记忆检索成功",
                            "result": {
                                "groups": [
                                    {
                                        "group_id": "group_456",
                                        "memories": [
                                            {
                                                "memory_type": "episode_summary",
                                                "user_id": "user_123",
                                                "timestamp": "2024-01-15T10:30:00",
                                                "summary": "讨论了咖啡偏好",
                                                "group_id": "group_456",
                                            }
                                        ],
                                        "scores": [0.95],
                                        "original_data": [],
                                    }
                                ],
                                "importance_scores": [0.85],
                                "total_count": 45,
                                "has_more": False,
                                "query_metadata": {
                                    "source": "episodic_memory_es_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve",
                                },
                                "metadata": {
                                    "source": "episodic_memory_es_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve",
                                },
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
                            "message": "query 不能为空",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve",
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
                            "message": "检索记忆失败，请稍后重试",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve",
                        }
                    }
                },
            },
        },
    )
    async def retrieve_memories(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        检索相关记忆数据

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 记忆检索响应

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 从请求中获取 JSON body
            body = await fastapi_request.json()
            query = body.get("query")
            logger.info(
                "收到 V3 retrieve 请求: user_id=%s, query=%s", body.get("user_id"), query
            )

            # 直接使用 converter 转换
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)

            # 使用 retrieve_mem 方法（支持 keyword 和 hybrid）
            response = await self.memory_manager.retrieve_mem(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "V3 retrieve 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"记忆检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("V3 retrieve 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 retrieve 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="检索记忆失败，请稍后重试"
            ) from e

    @post(
        "/retrieve_keyword",
        response_model=Dict[str, Any],
        summary="关键词检索相关记忆",
        description="""
        基于关键词（BM25）检索相关的记忆数据
        
        ## 功能说明：
        - 使用关键词匹配和 BM25 算法进行检索
        - 支持中文分词和停用词过滤
        - 支持时间范围过滤
        - 返回结果按群组组织，并包含相关性评分
        - 适用于需要精确匹配的场景
        
        ## 关键词检索优势：
        - **速度快**: 基于倒排索引，检索速度极快
        - **精确匹配**: 能够精确匹配查询中的关键词
        - **可解释性强**: 匹配结果直观，容易理解
        - **资源消耗低**: 不需要向量计算，资源消耗较小
        
        ## 与向量检索的对比：
        - 关键词检索（keyword）: 更快速、精确匹配，适合已知具体术语的查询
        - 向量检索（vector）: 更智能、语义理解，适合需要理解上下文的查询
        
        ## 返回结果说明：
        - 记忆按群组（group）组织返回
        - 每个群组包含多条相关记忆，按时间排序
        - 群组按重要性得分排序，最重要的群组排在前面
        - 每条记忆都有相关性得分，表示与查询的匹配程度
        
        ## 使用场景：
        - 精确关键词搜索
        - 已知术语或名称的查询
        - 需要快速响应的场景
        - 关键词高亮显示
        """,
        responses={
            200: {
                "description": "成功检索记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.OK.value,
                            "message": "关键词检索成功",
                            "result": {
                                "groups": [
                                    {
                                        "group_id": "group_456",
                                        "memories": [
                                            {
                                                "memory_type": "episode_summary",
                                                "user_id": "user_123",
                                                "timestamp": "2024-01-15T10:30:00",
                                                "summary": "讨论了咖啡偏好",
                                                "group_id": "group_456",
                                            }
                                        ],
                                        "scores": [0.95],
                                        "original_data": [],
                                    }
                                ],
                                "importance_scores": [0.85],
                                "total_count": 45,
                                "has_more": False,
                                "query_metadata": {
                                    "source": "episodic_memory_es_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve_keyword",
                                },
                                "metadata": {
                                    "source": "episodic_memory_es_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve_keyword",
                                },
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
                            "message": "query 不能为空",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve_keyword",
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
                            "message": "关键词检索失败，请稍后重试",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve_keyword",
                        }
                    }
                },
            },
        },
    )
    async def retrieve_memories_keyword(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        使用关键词（BM25）检索相关记忆数据

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 关键词记忆检索响应

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 从请求中获取 JSON body
            body = await fastapi_request.json()
            query = body.get("query")
            logger.info(
                "收到 V3 retrieve_keyword 请求: user_id=%s, query=%s",
                body.get("user_id"),
                query,
            )

            # 使用 converter 转换
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)
            retrieve_request.retrieve_method = RetrieveMethod.KEYWORD

            # 调用 memory_manager 的 retrieve_mem_keyword 方法
            response = await self.memory_manager.retrieve_mem_keyword(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "V3 retrieve_keyword 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"关键词检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("V3 retrieve_keyword 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 retrieve_keyword 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="关键词检索失败，请稍后重试"
            ) from e

    @post(
        "/retrieve_vector",
        response_model=Dict[str, Any],
        summary="向量检索相关记忆",
        description="""
        基于语义向量相似度检索相关的记忆数据
        
        ## 功能说明：
        - 将查询文本转换为向量嵌入（embedding）
        - 使用向量相似度（如余弦相似度）进行语义检索
        - 支持时间范围过滤
        - 返回结果按群组组织，并包含相似度评分
        - 适用于需要理解语义相关性的场景
        
        ## 向量检索优势：
        - **语义理解**: 能够理解查询的语义含义，而不仅仅是关键词匹配
        - **同义词识别**: 可以找到使用不同词汇但表达相同意思的记忆
        - **模糊匹配**: 适合查询意图不够明确的场景
        - **跨语言能力**: 某些模型支持跨语言的语义检索
        
        ## 与关键词检索的对比：
        - 关键词检索（keyword）: 更快速、精确匹配，适合已知具体术语的查询
        - 向量检索（vector）: 更智能、语义理解，适合需要理解上下文的查询
        
        ## 返回结果说明：
        - 记忆按群组（group）组织返回
        - 每个群组包含多条相关记忆，按时间排序
        - 群组按重要性得分排序，最重要的群组排在前面
        - 每条记忆都有相似度得分（0-1之间），分数越高表示越相关
        
        ## 使用场景：
        - 智能问答系统
        - 语义搜索引擎
        - 上下文理解和推理
        - 相似内容推荐
        """,
        responses={
            200: {
                "description": "成功检索记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.OK.value,
                            "message": "向量检索成功",
                            "result": {
                                "groups": [
                                    {
                                        "group_id": "group_456",
                                        "memories": [
                                            {
                                                "memory_type": "episode_summary",
                                                "user_id": "user_123",
                                                "timestamp": "2024-01-15T10:30:00",
                                                "summary": "讨论了咖啡偏好",
                                                "group_id": "group_456",
                                            }
                                        ],
                                        "scores": [0.95],
                                        "original_data": [],
                                    }
                                ],
                                "importance_scores": [0.85],
                                "total_count": 45,
                                "has_more": False,
                                "query_metadata": {
                                    "source": "episodic_memory_milvus_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve_vector",
                                },
                                "metadata": {
                                    "source": "episodic_memory_milvus_repository",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve_vector",
                                },
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
                            "message": "query 不能为空",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve_vector",
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
                            "message": "向量检索失败，请稍后重试",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve_vector",
                        }
                    }
                },
            },
        },
    )
    async def retrieve_memories_vector(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        使用向量相似度检索相关记忆数据

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 向量记忆检索响应

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 从请求中获取 JSON body
            body = await fastapi_request.json()
            query = body.get("query")
            logger.info(
                "收到 V3 retrieve_vector 请求: user_id=%s, query=%s",
                body.get("user_id"),
                query,
            )

            # 使用 converter 转换
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)
            retrieve_request.retrieve_method = RetrieveMethod.VECTOR

            # 调用 memory_manager 的 retrieve_mem_vector 方法
            response = await self.memory_manager.retrieve_mem_vector(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "V3 retrieve_vector 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"向量检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("V3 retrieve_vector 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 retrieve_vector 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="向量检索失败，请稍后重试"
            ) from e

    @post(
        "/retrieve_hybrid",
        response_model=Dict[str, Any],
        summary="混合检索相关记忆",
        description="""
        结合关键词检索和向量检索的混合方法检索相关记忆数据
        
        ## 功能说明：
        - 同时使用关键词（BM25）和向量相似度进行检索
        - 结合两种检索方法的优势，提供更准确的结果
        - 支持时间范围过滤
        - 返回结果按群组组织，并包含综合评分
        - 适用于需要高精度检索的场景
        
        ## 混合检索优势：
        - **精确匹配**: 关键词检索确保精确术语匹配
        - **语义理解**: 向量检索提供语义相关性理解
        - **互补性**: 两种方法相互补充，减少漏检和误检
        - **平衡性**: 在速度和准确性之间取得平衡
        
        ## 与其他检索方法的对比：
        - 关键词检索（keyword）: 快速、精确，但可能遗漏语义相关的内容
        - 向量检索（vector）: 智能、语义理解，但可能匹配不相关的同义词
        - 混合检索（hybrid）: 结合两者优势，提供最全面的检索结果
        
        ## 返回结果说明：
        - 记忆按群组（group）组织返回
        - 每个群组包含多条相关记忆，按时间排序
        - 群组按重要性得分排序，最重要的群组排在前面
        - 每条记忆都有综合得分，结合了关键词和向量相似度
        
        ## 使用场景：
        - 高精度智能问答系统
        - 专业文档检索
        - 复杂查询处理
        - 需要兼顾精确性和语义理解的场景
        """,
        responses={
            200: {
                "description": "成功检索记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.OK.value,
                            "message": "混合检索成功",
                            "result": {
                                "groups": [
                                    {
                                        "group_id": "group_456",
                                        "memories": [
                                            {
                                                "memory_type": "episode_summary",
                                                "user_id": "user_123",
                                                "timestamp": "2024-01-15T10:30:00",
                                                "summary": "讨论了咖啡偏好",
                                                "group_id": "group_456",
                                            }
                                        ],
                                        "scores": [0.95],
                                        "original_data": [],
                                    }
                                ],
                                "importance_scores": [0.85],
                                "total_count": 45,
                                "has_more": False,
                                "query_metadata": {
                                    "source": "retrieve_mem_hybrid_service",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve_hybrid",
                                },
                                "metadata": {
                                    "source": "retrieve_mem_hybrid_service",
                                    "user_id": "user_123",
                                    "memory_type": "retrieve_hybrid",
                                },
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
                            "message": "query 不能为空",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve_hybrid",
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
                            "message": "混合检索失败，请稍后重试",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v3/agentic/retrieve_hybrid",
                        }
                    }
                },
            },
        },
    )
    async def retrieve_memories_hybrid(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        使用混合方法检索相关记忆数据

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 混合记忆检索响应

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 从请求中获取 JSON body
            body = await fastapi_request.json()
            query = body.get("query")
            logger.info(
                "收到 V3 retrieve_hybrid 请求: user_id=%s, query=%s",
                body.get("user_id"),
                query,
            )

            # 使用 converter 转换
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)
            retrieve_request.retrieve_method = RetrieveMethod.HYBRID

            # 调用 memory_manager 的 retrieve_mem_hybrid 方法
            response = await self.memory_manager.retrieve_mem_hybrid(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "V3 retrieve_hybrid 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"混合检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("V3 retrieve_hybrid 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("V3 retrieve_hybrid 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="混合检索失败，请稍后重试"
            ) from e
