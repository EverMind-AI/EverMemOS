"""
Agentic Layer V2 控制器

提供 agentic layer 的 RESTful API 路由，每个功能使用独立的路由端点
"""

import logging
from typing import Any, Dict
from fastapi import HTTPException, Request as FastAPIRequest

from agentic_layer.schemas import RetrieveMethod
from core.di.decorators import controller
from core.interface.controller.base_controller import BaseController, post
from agentic_layer.memory_manager import MemoryManager
from agentic_layer.converter import (
    convert_dict_to_fetch_mem_request,
    convert_dict_to_retrieve_mem_request,
)
from agentic_layer.dtos.memory_query import RetrieveMemRequest, RetrieveMemRequest
from core.constants.errors import ErrorCode, ErrorStatus
from agentic_layer.converter import _handle_conversation_format

logger = logging.getLogger(__name__)


# ==================== 控制器实现 ====================


@controller("agentic_v2_controller", primary=True)
class AgenticV2Controller(BaseController):
    """
    Agentic Layer V2 API 控制器

    提供独立的路由端点用于不同的记忆操作：
    - 记忆存储 (memorize): 将原始数据存储为记忆
    - 记忆获取 (fetch): 使用 KV 方式获取用户核心记忆
    - 记忆检索 (retrieve): 支持关键词/向量/混合三种检索方法（通过 retrieve_method 参数控制）
      * keyword: 基于关键词的 BM25 检索
      * vector: 基于语义向量相似度检索
      * hybrid: 结合关键词和向量的混合检索
    - 关键词检索 (retrieve_keyword): 基于关键词的 BM25 检索（独立端点）
    - 向量检索 (retrieve_vector): 基于语义向量相似度检索相关记忆（独立端点）
    - 混合检索 (retrieve_hybrid): 结合关键词和向量检索的混合方法（独立端点）
    """

    def __init__(self):
        """初始化控制器"""
        super().__init__(
            prefix="/api/v2/agentic",
            tags=["Agentic Layer V2"],
            default_auth="none",  # 根据实际需求调整认证策略
        )
        self.memory_manager = MemoryManager()
        logger.info("AgenticV2Controller initialized with MemoryManager")

    @post(
        "/memorize",
        response_model=Dict[str, Any],
        summary="存储记忆数据",
        description="""
        将原始数据（如对话、邮件、文档等）存储为结构化记忆
        
        ## 功能说明：
        - 接收原始数据并提取记忆单元（memcells）
        - 自动识别数据类型：对话、邮件、文档等
        - 支持批量处理多条数据
        - 返回已保存的记忆列表
        
        ## 支持的数据类型：
        - **Conversation**: 对话消息，支持单条或多条消息
        - **Email**: 邮件数据
        - **LinkDoc**: 链接文档（Notion、Google Drive、Dropbox、Memo等）
        
        ## 请求格式说明：
        - **messages**: 对话消息列表格式（多条消息）
        - **raw_data_type**: 标准 RawData 格式
        - 其他格式请参考文档
        
        ## 使用场景：
        - 实时对话记忆存储
        - 邮件内容记忆化
        - 文档知识提取
        - 批量数据导入
        """,
        responses={
            200: {
                "description": "成功存储记忆数据",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "记忆存储成功",
                            "result": {
                                "saved_memories": [
                                    {
                                        "memory_type": "episode_summary",
                                        "user_id": "user_123",
                                        "group_id": "group_456",
                                        "timestamp": "2024-01-15T10:30:00",
                                        "content": "用户讨论了咖啡偏好",
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
                            "message": "数据格式错误，缺少必需字段",
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v2/agentic/memorize",
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
                            "timestamp": "2024-01-15T10:30:00+00:00",
                            "path": "/api/v2/agentic/memorize",
                        }
                    }
                },
            },
        },
    )
    async def memorize_data(self, fastapi_request: FastAPIRequest) -> Dict[str, Any]:
        """
        存储记忆数据

        Args:
            fastapi_request: FastAPI 请求对象

        Returns:
            Dict[str, Any]: 记忆存储响应，包含已保存的记忆列表

        Raises:
            HTTPException: 当请求处理失败时
        """
        try:
            # 从请求中获取 JSON body
            body = await fastapi_request.json()
            logger.info("收到 memorize 请求: %s", body)

            # 使用 converter 转换为 MemorizeRequest
            memorize_request = await _handle_conversation_format(body)
            # 调用 memory_manager 的 memorize 方法
            logger.info("开始处理记忆请求")
            memories = await self.memory_manager.memorize(memorize_request)

            # 返回统一格式的响应
            memory_count = len(memories) if memories else 0
            logger.info("处理记忆请求完成，保存了 %s 条记忆", memory_count)
            return {
                "status": ErrorStatus.OK.value,
                "message": f"记忆存储成功，共保存 {memory_count} 条记忆",
                "result": {"saved_memories": memories, "count": memory_count},
            }

        except ValueError as e:
            logger.error("memorize 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("memorize 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="存储记忆失败，请稍后重试"
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
                            "path": "/api/v2/agentic/fetch",
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
                            "path": "/api/v2/agentic/fetch",
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
                "收到 fetch 请求: user_id=%s, memory_type=%s",
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
                "fetch 请求处理完成: user_id=%s, 返回 %s 条记忆",
                body.get("user_id"),
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
        except Exception as e:
            logger.error("fetch 请求处理失败: %s", e, exc_info=True)
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
                            "path": "/api/v2/agentic/retrieve",
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
                            "path": "/api/v2/agentic/retrieve",
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
                "收到 retrieve 请求: user_id=%s, query=%s", body.get("user_id"), query
            )

            # 直接使用 converter 转换
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)

            # 使用 retrieve_mem 方法（支持 keyword 和 hybrid）
            response = await self.memory_manager.retrieve_mem(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "retrieve 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"记忆检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("retrieve 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("retrieve 请求处理失败: %s", e, exc_info=True)
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
                            "path": "/api/v2/agentic/retrieve_keyword",
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
                            "path": "/api/v2/agentic/retrieve_keyword",
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
                "收到 retrieve_keyword 请求: user_id=%s, query=%s",
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
                "retrieve_keyword 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"关键词检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("retrieve_keyword 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("retrieve_keyword 请求处理失败: %s", e, exc_info=True)
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
        - 使用向量相似度（COSINE）进行语义检索
        - 支持时间范围过滤和相似度阈值配置
        - 返回结果按群组组织，并包含相似度评分
        - 适用于需要理解语义相关性的场景
        
        ## 请求参数：
        - **radius** (可选): COSINE 相似度阈值，范围 [-1, 1]，默认 0.6
          * 只返回相似度 >= radius 的结果
          * 值越大，要求相似度越高，结果越精准但可能更少
        
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
                            "path": "/api/v2/agentic/retrieve_vector",
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
                            "path": "/api/v2/agentic/retrieve_vector",
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
                "收到 retrieve_vector 请求: user_id=%s, query=%s",
                body.get("user_id"),
                query,
            )

            # 使用 converter 转换（RetrieveMemRequest 继承自 RetrieveMemRequest）
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)
            retrieve_request.retrieve_method = RetrieveMethod.VECTOR

            # 调用 memory_manager 的 retrieve_mem_vector 方法
            response = await self.memory_manager.retrieve_mem_vector(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "retrieve_vector 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"向量检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("retrieve_vector 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("retrieve_vector 请求处理失败: %s", e, exc_info=True)
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
        - 同时使用关键词（BM25）和向量相似度（COSINE）进行检索
        - 结合两种检索方法的优势，提供更准确的结果
        - 支持时间范围过滤和相似度阈值配置
        - 返回结果按群组组织，并包含综合评分
        - 适用于需要高精度检索的场景
        
        ## 请求参数：
        - **radius** (可选): COSINE 相似度阈值，范围 [-1, 1]，默认 0.6
          * 只返回相似度 >= radius 的结果
          * 影响向量检索部分的结果质量
        
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
                            "path": "/api/v2/agentic/retrieve_hybrid",
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
                            "path": "/api/v2/agentic/retrieve_hybrid",
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
                "收到 retrieve_hybrid 请求: user_id=%s, query=%s",
                body.get("user_id"),
                query,
            )

            # 使用 converter 转换（RetrieveMemRequest 继承自 RetrieveMemRequest）
            retrieve_request = convert_dict_to_retrieve_mem_request(body, query=query)
            retrieve_request.retrieve_method = RetrieveMethod.HYBRID

            # 调用 memory_manager 的 retrieve_mem_hybrid 方法
            response = await self.memory_manager.retrieve_mem_hybrid(retrieve_request)

            # 返回统一格式的响应
            group_count = len(response.memories) if response.memories else 0
            logger.info(
                "retrieve_hybrid 请求处理完成: user_id=%s, 返回 %s 个群组",
                body.get("user_id"),
                group_count,
            )
            return {
                "status": ErrorStatus.OK.value,
                "message": f"混合检索成功，共检索到 {group_count} 个群组",
                "result": response,
            }

        except ValueError as e:
            logger.error("retrieve_hybrid 请求参数错误: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("retrieve_hybrid 请求处理失败: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500, detail="混合检索失败，请稍后重试"
            ) from e
