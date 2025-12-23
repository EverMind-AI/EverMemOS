# -*- coding: utf-8 -*-
"""
Memory API DTO

用于 Memory API 的请求和响应数据传输对象。
这些模型用于定义 OpenAPI 参数文档。
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class MemorizeMessageRequest(BaseModel):
    """
    存储单条消息请求体
    
    用于 POST /api/v1/memories 接口
    """
    
    group_id: Optional[str] = Field(
        default=None,
        description="群组 ID",
        examples=["group_123"]
    )
    group_name: Optional[str] = Field(
        default=None,
        description="群组名称",
        examples=["项目讨论群"]
    )
    message_id: str = Field(
        ...,
        description="消息唯一标识符",
        examples=["msg_001"]
    )
    create_time: str = Field(
        ...,
        description="消息创建时间（ISO 8601 格式）",
        examples=["2025-01-15T10:00:00+00:00"]
    )
    sender: str = Field(
        ...,
        description="发送者用户 ID",
        examples=["user_001"]
    )
    sender_name: Optional[str] = Field(
        default=None,
        description="发送者名称（若不提供则使用 sender）",
        examples=["张三"]
    )
    content: str = Field(
        ...,
        description="消息内容",
        examples=["今天我们来讨论一下新功能的技术方案"]
    )
    refer_list: Optional[List[str]] = Field(
        default=None,
        description="引用的消息 ID 列表",
        examples=[["msg_000"]]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "group_id": "group_123",
                "group_name": "项目讨论群",
                "message_id": "msg_001",
                "create_time": "2025-01-15T10:00:00+00:00",
                "sender": "user_001",
                "sender_name": "张三",
                "content": "今天我们来讨论一下新功能的技术方案",
                "refer_list": ["msg_000"]
            }
        }
    }


class FetchMemoriesParams(BaseModel):
    """
    获取用户记忆的查询参数
    
    用于 GET /api/v1/memories 接口
    """
    
    user_id: str = Field(
        ...,
        description="用户 ID",
        examples=["user_123"]
    )
    memory_type: Optional[str] = Field(
        default="multiple",
        description="记忆类型：profile（用户画像）、episode_memory（情节记忆）、foresight（前瞻记忆）、event_log（事件日志）、multiple（多种类型，默认）",
        examples=["profile"]
    )
    limit: Optional[int] = Field(
        default=10,
        description="返回记忆的最大数量",
        ge=1,
        le=100,
        examples=[20]
    )
    offset: Optional[int] = Field(
        default=0,
        description="分页偏移量",
        ge=0,
        examples=[0]
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="排序字段",
        examples=["created_at"]
    )
    sort_order: Optional[str] = Field(
        default="desc",
        description="排序方向：asc（升序）或 desc（降序）",
        examples=["desc"]
    )
    version_range: Optional[List[Optional[str]]] = Field(
        default=None,
        description="版本范围过滤，格式为 [start, end]，闭区间",
        examples=[["v1.0", "v2.0"]]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "memory_type": "profile",
                "limit": 20,
                "offset": 0,
                "sort_order": "desc"
            }
        }
    }


class SearchMemoriesRequest(BaseModel):
    """
    搜索记忆的请求参数
    
    用于 GET /api/v1/memories/search 接口
    支持通过 query params 或 body 传递参数
    """
    
    user_id: Optional[str] = Field(
        default=None,
        description="用户 ID（user_id 和 group_id 至少提供一个）",
        examples=["user_123"]
    )
    group_id: Optional[str] = Field(
        default=None,
        description="群组 ID（user_id 和 group_id 至少提供一个）",
        examples=["group_456"]
    )
    query: Optional[str] = Field(
        default=None,
        description="搜索查询文本",
        examples=["咖啡偏好"]
    )
    retrieve_method: Optional[str] = Field(
        default="keyword",
        description="检索方法：keyword（关键词，默认）、vector（向量）、hybrid（混合）、rrf（RRF融合）、agentic（智能检索）",
        examples=["keyword"]
    )
    top_k: Optional[int] = Field(
        default=10,
        description="返回结果的最大数量",
        ge=1,
        le=100,
        examples=[10]
    )
    memory_types: Optional[List[str]] = Field(
        default=None,
        description="要检索的记忆类型列表：episode_memory、foresight、event_log（不支持 profile）",
        examples=[["episode_memory"]]
    )
    start_time: Optional[str] = Field(
        default=None,
        description="时间范围起始（ISO 8601 格式）",
        examples=["2024-01-01T00:00:00"]
    )
    end_time: Optional[str] = Field(
        default=None,
        description="时间范围结束（ISO 8601 格式）",
        examples=["2024-12-31T23:59:59"]
    )
    radius: Optional[float] = Field(
        default=None,
        description="向量检索的 COSINE 相似度阈值（仅用于 vector 和 hybrid 方法，默认 0.6）",
        ge=0.0,
        le=1.0,
        examples=[0.6]
    )
    include_metadata: Optional[bool] = Field(
        default=True,
        description="是否包含元数据",
        examples=[True]
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外的过滤条件",
        examples=[{}]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "query": "咖啡偏好",
                "retrieve_method": "keyword",
                "top_k": 10,
                "memory_types": ["episode_memory"]
            }
        }
    }


class UserDetailRequest(BaseModel):
    """用户详情请求模型"""
    
    full_name: str = Field(
        ...,
        description="用户全名",
        examples=["张三"]
    )
    role: Optional[str] = Field(
        default=None,
        description="用户角色",
        examples=["developer"]
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外信息",
        examples=[{"department": "Engineering"}]
    )


class ConversationMetaCreateRequest(BaseModel):
    """
    保存会话元数据请求体
    
    用于 POST /api/v1/memories/conversation-meta 接口
    """
    
    version: str = Field(
        ...,
        description="元数据版本号",
        examples=["1.0"]
    )
    scene: str = Field(
        ...,
        description="场景标识符",
        examples=["group_chat"]
    )
    scene_desc: Dict[str, Any] = Field(
        ...,
        description="场景描述对象，可包含 bot_ids 等字段",
        examples=[{"bot_ids": ["bot_001"], "type": "project_discussion"}]
    )
    name: str = Field(
        ...,
        description="会话名称",
        examples=["项目讨论群"]
    )
    description: Optional[str] = Field(
        default=None,
        description="会话描述",
        examples=["新功能开发技术讨论"]
    )
    group_id: str = Field(
        ...,
        description="群组唯一标识符",
        examples=["group_123"]
    )
    created_at: str = Field(
        ...,
        description="会话创建时间（ISO 8601 格式）",
        examples=["2025-01-15T10:00:00+00:00"]
    )
    default_timezone: Optional[str] = Field(
        default=None,
        description="默认时区",
        examples=["UTC"]
    )
    user_details: Optional[Dict[str, UserDetailRequest]] = Field(
        default=None,
        description="参与者详情，key 为用户 ID，value 为用户详情对象",
        examples=[{
            "user_001": {"full_name": "张三", "role": "developer", "extra": {"department": "Engineering"}},
            "user_002": {"full_name": "李四", "role": "designer", "extra": {"department": "Design"}}
        }]
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="标签列表",
        examples=[["work", "technical"]]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "version": "1.0",
                "scene": "group_chat",
                "scene_desc": {"bot_ids": ["bot_001"], "type": "project_discussion"},
                "name": "项目讨论群",
                "description": "新功能开发技术讨论",
                "group_id": "group_123",
                "created_at": "2025-01-15T10:00:00+00:00",
                "default_timezone": "UTC",
                "user_details": {
                    "user_001": {"full_name": "张三", "role": "developer", "extra": {"department": "Engineering"}}
                },
                "tags": ["work", "technical"]
            }
        }
    }


class ConversationMetaPatchRequest(BaseModel):
    """
    部分更新会话元数据请求体
    
    用于 PATCH /api/v1/memories/conversation-meta 接口
    """
    
    group_id: str = Field(
        ...,
        description="要更新的群组 ID（必填）",
        examples=["group_123"]
    )
    name: Optional[str] = Field(
        default=None,
        description="新的会话名称",
        examples=["新会话名称"]
    )
    description: Optional[str] = Field(
        default=None,
        description="新的会话描述",
        examples=["更新后的描述"]
    )
    scene_desc: Optional[Dict[str, Any]] = Field(
        default=None,
        description="新的场景描述",
        examples=[{"bot_ids": ["bot_002"]}]
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="新的标签列表",
        examples=[["tag1", "tag2"]]
    )
    user_details: Optional[Dict[str, UserDetailRequest]] = Field(
        default=None,
        description="新的用户详情（将完全替换现有的 user_details）",
        examples=[{"user_001": {"full_name": "张三", "role": "lead"}}]
    )
    default_timezone: Optional[str] = Field(
        default=None,
        description="新的默认时区",
        examples=["Asia/Shanghai"]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "group_id": "group_123",
                "name": "新会话名称",
                "tags": ["updated", "tags"]
            }
        }
    }

