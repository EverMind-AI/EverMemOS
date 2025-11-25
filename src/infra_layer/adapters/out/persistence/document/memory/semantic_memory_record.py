"""
SemanticMemoryRecord Beanie ODM 模型

统一存储从情景记忆（个人或群组）中提取的语义记忆。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase
from beanie import PydanticObjectId


class SemanticMemoryRecord(DocumentBase, AuditBase):
    """
    通用语义记忆文档模型

    统一存储个人或群组情景记忆中提取的语义信息。
    当 user_id 存在时代表个人语义记忆；当 user_id 为空而 group_id 存在时代表群组语义记忆。
    """

    # 核心字段
    user_id: Optional[str] = Field(
        default=None, description="用户ID，个人记忆必填，群组记忆为None"
    )
    user_name: Optional[str] = Field(default=None, description="用户名称")
    group_id: Optional[str] = Field(default=None, description="群组ID")
    group_name: Optional[str] = Field(default=None, description="群组名称")
    content: str = Field(..., min_length=1, description="语义记忆内容")
    parent_episode_id: str = Field(..., description="父情景记忆的 event_id")

    # 时间范围字段
    start_time: Optional[str] = Field(
        default=None, description="语义记忆开始时间（日期字符串，如 2024-01-01）"
    )
    end_time: Optional[str] = Field(
        default=None, description="语义记忆结束时间（日期字符串，如 2024-12-31）"
    )
    duration_days: Optional[int] = Field(default=None, description="持续天数")

    # 群组和参与者信息
    participants: Optional[List[str]] = Field(default=None, description="相关参与者")

    # 向量和模型
    vector: Optional[List[float]] = Field(
        default=None, description="语义记忆的文本向量"
    )
    vector_model: Optional[str] = Field(default=None, description="使用的向量化模型")

    # 证据和扩展信息
    evidence: Optional[str] = Field(default=None, description="支持该语义记忆的证据")
    extend: Optional[Dict[str, Any]] = Field(default=None, description="扩展字段")

    model_config = ConfigDict(
        collection="semantic_memory_records",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "id": "semantic_memory_001",
                "user_id": "user_12345",
                "user_name": "Alice",
                "content": "用户喜欢吃川菜，尤其是麻辣火锅",
                "parent_episode_id": "episode_001",
                "start_time": "2024-01-01",
                "end_time": "2024-12-31",
                "duration_days": 365,
                "group_id": "group_friends",
                "group_name": "朋友群",
                "participants": ["张三", "李四"],
                "vector": [0.1, 0.2, 0.3],
                "vector_model": "text-embedding-3-small",
                "evidence": "多次在聊天中提到喜欢吃火锅",
                "extend": {"confidence": 0.9},
            }
        },
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """兼容性属性，返回文档ID"""
        return self.id

    class Settings:
        """Beanie 设置"""

        name = "semantic_memory_records"

        indexes = [
            # 用户ID索引
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            # 父情景记忆索引
            IndexModel([("parent_episode_id", ASCENDING)], name="idx_parent_episode"),
            # 用户ID和父情景记忆复合索引
            IndexModel(
                [("user_id", ASCENDING), ("parent_episode_id", ASCENDING)],
                name="idx_user_parent",
            ),
            # 群组ID索引
            IndexModel([("group_id", ASCENDING)], name="idx_group_id", sparse=True),
            # 创建时间索引
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            # 更新时间索引
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]

        validate_on_save = True
        use_state_management = True


class SemanticMemoryRecordProjection(DocumentBase, AuditBase):
    """
    语义记忆简化模型（不包含向量）

    用于大部分不需要向量数据的场景，减少数据传输和内存占用。
    """

    # 核心字段
    id: Optional[PydanticObjectId] = Field(default=None, description="记录ID")
    user_id: Optional[str] = Field(
        default=None, description="用户ID，个人记忆必填，群组记忆为None"
    )
    user_name: Optional[str] = Field(default=None, description="用户名称")
    group_id: Optional[str] = Field(default=None, description="群组ID")
    group_name: Optional[str] = Field(default=None, description="群组名称")
    content: str = Field(..., min_length=1, description="语义记忆内容")
    parent_episode_id: str = Field(..., description="父情景记忆的 event_id")

    # 时间范围字段
    start_time: Optional[str] = Field(
        default=None, description="语义记忆开始时间（日期字符串，如 2024-01-01）"
    )
    end_time: Optional[str] = Field(
        default=None, description="语义记忆结束时间（日期字符串，如 2024-12-31）"
    )
    duration_days: Optional[int] = Field(default=None, description="持续天数")

    # 群组和参与者信息
    participants: Optional[List[str]] = Field(default=None, description="相关参与者")

    # 向量模型信息（保留模型名称，但不包含向量数据）
    vector_model: Optional[str] = Field(default=None, description="使用的向量化模型")

    # 证据和扩展信息
    evidence: Optional[str] = Field(default=None, description="支持该语义记忆的证据")
    extend: Optional[Dict[str, Any]] = Field(default=None, description="扩展字段")

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat(), PydanticObjectId: str},
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """兼容性属性，返回文档ID"""
        return self.id


# 导出模型
__all__ = ["SemanticMemoryRecord", "SemanticMemoryRecordProjection"]
