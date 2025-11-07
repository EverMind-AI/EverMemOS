"""
Agentic Layer V3 控制器

提供专门用于处理群聊记忆的 RESTful API 路由
直接接收简单直接的消息格式，逐条处理并存储
"""

import logging
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from core.di.decorators import controller
from core.di import get_bean_by_type
from core.interface.controller.base_controller import BaseController, post
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.memory_manager import MemoryManager
from agentic_layer.converter import _handle_conversation_format
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

    提供专门用于群聊记忆存储的接口：
    - memorize: 逐条接收简单直接的单条消息并存储为记忆
    """

    def __init__(self,conversation_meta_repository: ConversationMetaRawRepository):
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
