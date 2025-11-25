"""
Memory Controller - 统一的记忆管理控制器

提供完整的记忆管理 RESTful API 路由，包括：
- 记忆存储 (memorize): 接收简单直接的单条消息格式并存储
- 对话元数据 (conversation-meta): 保存对话的元数据信息
- 记忆获取 (fetch): 通过 GET 方法获取用户核心记忆
- 记忆检索 (search): 支持关键词/向量/混合三种检索方法
"""

import logging
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import BaseController, get, post, patch
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.memory_manager import MemoryManager
from api_specs.converter import (
    _handle_conversation_format,
    convert_dict_to_fetch_mem_request,
    convert_dict_to_retrieve_mem_request,
)
from api_specs.dtos.memory_query import ConversationMetaRequest, UserDetail
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

logger = logging.getLogger(__name__)


@controller("memory_controller", primary=True)
class MemoryController(BaseController):
    """
    Memory Controller
    """

    def __init__(self, conversation_meta_repository: ConversationMetaRawRepository):
        """初始化控制器"""
        super().__init__(
            prefix="/api/v1/memories",
            tags=["Memory Controller"],
            default_auth="none",  # 根据实际需求调整认证策略
        )
        self.memory_manager = MemoryManager()
        self.conversation_meta_repository = conversation_meta_repository
        # 获取 RedisProvider
        self.redis_provider = get_bean_by_type(RedisProvider)
        logger.info(
            "MemoryController initialized with MemoryManager and ConversationMetaRepository"
        )

    @post(
        "",
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
        
        ## 接口说明：
        - 接收简单直接的单条消息格式，无需转换
        
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
                        "examples": {
                            "extracted": {
                                "summary": "提取到记忆（已触发边界）",
                                "value": {
                                    "status": "ok",
                                    "message": "Extracted 1 memories",
                                    "result": {
                                        "saved_memories": [
                                            {
                                                "memory_type": "episode_memory",
                                                "user_id": "user_001",
                                                "group_id": "group_123",
                                                "timestamp": "2025-01-15T10:00:00",
                                                "content": "用户讨论了新功能的技术方案",
                                            }
                                        ],
                                        "count": 1,
                                        "status_info": "extracted",
                                    },
                                },
                            },
                            "accumulated": {
                                "summary": "消息已累积（未触发边界）",
                                "value": {
                                    "status": "ok",
                                    "message": "Message queued, awaiting boundary detection",
                                    "result": {
                                        "saved_memories": [],
                                        "count": 0,
                                        "status_info": "accumulated",
                                    },
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
                            "message": "数据格式错误：缺少必需字段 message_id",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories",
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
                            "path": "/api/v1/memories",
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
            logger.info("收到 memorize 请求（单条消息）")

            # 3. 使用 group_chat_converter 转换为内部格式
            logger.info("开始转换简单消息格式到内部格式")
            memorize_input = convert_simple_message_to_memorize_input(message_data)

            # 提取元信息用于日志
            group_name = memorize_input.get("group_name")
            group_id = memorize_input.get("group_id")

            logger.info("转换完成: group_id=%s, group_name=%s", group_id, group_name)

            # 4. 转换为 MemorizeRequest 对象并调用 memory_manager
            logger.info("开始处理记忆请求")
            memorize_request = await _handle_conversation_format(memorize_input)
            memories = await self.memory_manager.memorize(memorize_request)

            # 5. 返回统一格式的响应
            # 确保 saved_memories 始终返回列表（空列表或有数据的列表），而不是 None
            saved_memories = memories if memories else []
            memory_count = len(saved_memories)
            logger.info("处理记忆请求完成，保存了 %s 条记忆", memory_count)

            # 优化返回信息，帮助用户理解运行状态
            if memory_count > 0:
                message = f"Extracted {memory_count} memories"
            else:
                message = "Message queued, awaiting boundary detection"

            return {
                "status": ErrorStatus.OK.value,
                "message": message,
                "result": {
                    "saved_memories": saved_memories,
                    "count": memory_count,
                    "status_info": "accumulated" if memory_count == 0 else "extracted",
                },
            }

        except ValueError as e:
            logger.error("memorize 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("memorize 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="存储记忆失败，请稍后重试"
            ) from e

    @get(
        "",
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
        - **episode_memory**: 情景记忆摘要
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
                            "path": "/api/v1/memories/fetch",
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
                            "path": "/api/v1/memories/fetch",
                        }
                    }
                },
            },
        },
    )
    async def fetch_memories(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        获取用户记忆数据

        通过 KV 方式根据用户ID直接获取存储的核心记忆

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 记忆获取响应

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 从 query parameters 获取参数
            query_params = dict(fastapi_request.query_params)
            logger.info(
                "收到 fetch 请求: user_id=%s, memory_type=%s",
                query_params.get("user_id"),
                query_params.get("memory_type"),
            )

            # 直接使用 converter 转换
            fetch_request = convert_dict_to_fetch_mem_request(query_params)

            # 调用 memory_manager 的 fetch_mem 方法
            response = await self.memory_manager.fetch_mem(fetch_request)

            # 返回统一格式的响应
            memory_count = len(response.memories) if response.memories else 0
            logger.info(
                "fetch 请求处理完成: user_id=%s, 返回 %s 条记忆",
                query_params.get("user_id"),
                memory_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"记忆获取成功，共获取 {memory_count} 条记忆",
                "result": response,
            }

        except ValueError as e:
            logger.error("fetch 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("fetch 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="获取记忆失败，请稍后重试"
            ) from e

    @get(
        "/search",
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
                                                "memory_type": "episode_memory",
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
                            "path": "/api/v1/memories/search",
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
                            "path": "/api/v1/memories/search",
                        }
                    }
                },
            },
        },
    )
    async def search_memories(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        检索相关记忆数据

        基于查询文本使用关键词、向量或混合方法检索相关的记忆数据

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
                "收到 search 请求: user_id=%s, query=%s", body.get("user_id"), query
            )

            # 直接使用 converter 转换
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)

            # 使用 retrieve_mem 方法（支持 keyword 和 hybrid）
            response = await self.memory_manager.retrieve_mem(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "search 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"记忆检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("search 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("search 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="检索记忆失败，请稍后重试"
            ) from e

    @post(
        "/conversation-meta",
        response_model=Dict[str, Any],
        summary="保存对话元数据",
        description="""
        保存对话的元数据信息，包括场景、参与者、标签等
        
        ## 功能说明：
        - 如果 group_id 已存在，则更新整个记录（upsert）
        - 如果 group_id 不存在，则创建新记录
        - 所有字段都需要提供完整数据
        
        ## 注意事项：
        - 这是一个完整更新接口，会替换整个记录
        - 如果只需要更新部分字段，请使用 PATCH /conversation-meta 接口
        """,
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
            logger.info(
                "收到 conversation-meta 保存请求: group_id=%s",
                request_data.get("group_id"),
            )

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

            logger.info(
                "解析 ConversationMetaRequest 成功: group_id=%s",
                conversation_meta_request.group_id,
            )

            # 3. 转换为 ConversationMeta ODM 模型
            user_details_model = {}
            for user_id, detail in conversation_meta_request.user_details.items():
                user_details_model[user_id] = UserDetailModel(
                    full_name=detail.full_name, role=detail.role, extra=detail.extra
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
                raise HTTPException(status_code=500, detail="保存对话元数据失败")

            logger.info(
                "保存对话元数据成功: id=%s, group_id=%s",
                saved_meta.id,
                saved_meta.group_id,
            )

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
                    "created_at": (
                        saved_meta.created_at.isoformat()
                        if saved_meta.created_at
                        else None
                    ),
                    "updated_at": (
                        saved_meta.updated_at.isoformat()
                        if saved_meta.updated_at
                        else None
                    ),
                },
            }

        except KeyError as e:
            logger.error("conversation-meta 请求缺少必需字段: %s", e)
            raise HTTPException(
                status_code=400, detail=f"缺少必需字段: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("conversation-meta 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except Exception as e:
            logger.error("conversation-meta 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="保存对话元数据失败，请稍后重试"
            ) from e

    @patch(
        "/conversation-meta",
        response_model=Dict[str, Any],
        summary="局部更新对话元数据",
        description="""
        局部更新对话的元数据信息，只更新提供的字段
        
        ## 功能说明：
        - 根据 group_id 定位要更新的对话元数据
        - 只更新请求中提供的字段，未提供的字段保持不变
        - 适用于需要修改部分信息的场景
        
        ## 请求示例：
        ```json
        {
          "group_id": "group_123",
          "name": "新的对话名称",
          "tags": ["标签1", "标签2"]
        }
        ```
        
        ## 字段说明：
        - **group_id** (必需): 要更新的对话群组ID
        - 其他所有字段均为可选，只需提供需要更新的字段
        
        ## 支持更新的字段：
        - **name**: 对话名称
        - **description**: 对话描述
        - **scene_desc**: 场景描述
        - **tags**: 标签列表
        - **user_details**: 用户详情（会完整替换现有的 user_details）
        - **default_timezone**: 默认时区
        
        ## 注意事项：
        - group_id 必须存在，否则返回 404 错误
        - user_details 字段如果提供，会完整替换现有的用户详情
        - 不允许修改 version、scene、group_id、conversation_created_at 等核心字段
        """,
        responses={
            200: {
                "description": "成功更新对话元数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "对话元数据更新成功",
                            "result": {
                                "id": "507f1f77bcf86cd799439011",
                                "group_id": "group_123",
                                "name": "新的对话名称",
                                "updated_fields": ["name", "tags"],
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
                            "message": "缺少必需字段 group_id",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
            404: {
                "description": "对话元数据不存在",
                "content": {
                    "application/json": {
                        "example": {
                            "status": ErrorStatus.FAILED.value,
                            "code": ErrorCode.RESOURCE_NOT_FOUND.value,
                            "message": "找不到指定的对话元数据: group_123",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
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
                            "message": "更新对话元数据失败，请稍后重试",
                            "timestamp": "2025-01-15T10:30:00+00:00",
                            "path": "/api/v1/memories/conversation-meta",
                        }
                    }
                },
            },
        },
    )
    async def patch_conversation_meta(
        self, fastapi_request: FastAPIRequest
    ) -> Dict[str, Any]:
        """
        局部更新对话元数据

        根据 group_id 定位记录，只更新请求中提供的字段

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 更新响应，包含更新后的元数据信息

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 1. 从请求中获取 JSON body
            request_data = await fastapi_request.json()
            group_id = request_data.get("group_id")

            # 2. 验证 group_id 是否提供
            if not group_id:
                raise HTTPException(status_code=400, detail="缺少必需字段 group_id")

            logger.info("收到 conversation-meta 局部更新请求: group_id=%s", group_id)

            # 3. 检查对话元数据是否存在
            existing_meta = await self.conversation_meta_repository.get_by_group_id(
                group_id
            )
            if not existing_meta:
                raise HTTPException(
                    status_code=404, detail=f"找不到指定的对话元数据: {group_id}"
                )

            # 4. 准备更新数据（排除不允许修改的字段和空值）
            # 不允许通过 PATCH 修改的核心字段
            immutable_fields = {
                "version",
                "scene",
                "group_id",
                "conversation_created_at",
            }

            update_data = {}
            updated_fields = []

            # 处理普通字段
            for key, value in request_data.items():
                if key in immutable_fields or key == "group_id":
                    continue

                # 处理 user_details 字段
                if key == "user_details" and value is not None:
                    user_details_model = {}
                    for user_id, detail_data in value.items():
                        user_details_model[user_id] = UserDetailModel(
                            full_name=detail_data["full_name"],
                            role=detail_data["role"],
                            extra=detail_data.get("extra", {}),
                        )
                    update_data["user_details"] = user_details_model
                    updated_fields.append(key)
                elif value is not None:
                    update_data[key] = value
                    updated_fields.append(key)

            # 5. 如果没有需要更新的字段
            if not update_data:
                logger.warning("没有提供需要更新的字段: group_id=%s", group_id)
                return {
                    "status": ErrorStatus.OK.value,
                    "message": "没有字段需要更新",
                    "result": {
                        "id": str(existing_meta.id),
                        "group_id": existing_meta.group_id,
                        "updated_fields": [],
                    },
                }

            # 6. 执行更新
            logger.info("开始更新对话元数据，更新字段: %s", updated_fields)
            updated_meta = await self.conversation_meta_repository.update_by_group_id(
                group_id=group_id, update_data=update_data
            )

            if not updated_meta:
                raise HTTPException(status_code=500, detail="更新对话元数据失败")

            logger.info(
                "更新对话元数据成功: id=%s, group_id=%s, updated_fields=%s",
                updated_meta.id,
                updated_meta.group_id,
                updated_fields,
            )

            # 7. 返回成功响应
            return {
                "status": ErrorStatus.OK.value,
                "message": f"对话元数据更新成功，共更新 {len(updated_fields)} 个字段",
                "result": {
                    "id": str(updated_meta.id),
                    "group_id": updated_meta.group_id,
                    "scene": updated_meta.scene,
                    "name": updated_meta.name,
                    "updated_fields": updated_fields,
                    "updated_at": (
                        updated_meta.updated_at.isoformat()
                        if updated_meta.updated_at
                        else None
                    ),
                },
            }

        except HTTPException:
            # 重新抛出 HTTPException
            raise
        except KeyError as e:
            logger.error("conversation-meta 局部更新请求缺少必需字段: %s", e)
            raise HTTPException(
                status_code=400, detail=f"缺少必需字段: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("conversation-meta 局部更新请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("conversation-meta 局部更新请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="更新对话元数据失败，请稍后重试"
            ) from e
