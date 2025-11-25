"""
EventLogRecord Beanie ODM 模型

统一存储从情景记忆（个人或群组）中提取的事件日志（原子事实）。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase
from beanie import PydanticObjectId


class EventLogRecord(DocumentBase, AuditBase):
    """
    通用事件日志文档模型

    统一存储个人或群组情景记忆中拆分出的原子事实，用于细粒度检索。
    """

    # 核心字段
    user_id: Optional[str] = Field(default=None, description="用户ID，个人事件必填")
    user_name: Optional[str] = Field(default=None, description="用户名称")
    group_id: Optional[str] = Field(default=None, description="群组ID")
    group_name: Optional[str] = Field(default=None, description="群组名称")
    atomic_fact: str = Field(..., description="原子事实内容（单条句子）")
    parent_episode_id: str = Field(..., description="父情景记忆的 event_id")

    # 时间信息
    timestamp: datetime = Field(..., description="事件发生时间")

    # 群组和参与者信息
    participants: Optional[List[str]] = Field(default=None, description="相关参与者")

    # 向量和模型
    vector: Optional[List[float]] = Field(default=None, description="原子事实向量")
    vector_model: Optional[str] = Field(default=None, description="使用的向量化模型")

    # 事件类型和扩展信息
    event_type: Optional[str] = Field(
        default=None, description="事件类型，如 Conversation"
    )
    extend: Optional[Dict[str, Any]] = Field(default=None, description="扩展字段")

    model_config = ConfigDict(
        collection="event_log_records",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "id": "atomic_fact_001",
                "user_id": "user_12345",
                "user_name": "Alice",
                "atomic_fact": "用户在2024年1月1日去了成都，喜欢当地的川菜。",
                "parent_episode_id": "episode_001",
                "timestamp": "2024-01-01T10:00:00+08:00",
                "group_id": "group_travel",
                "group_name": "旅行群",
                "participants": ["张三", "李四"],
                "vector": [0.1, 0.2, 0.3],
                "event_type": "Conversation",
                "extend": {"location": "成都"},
            }
        },
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """兼容性属性，返回文档ID"""
        return self.id

    class Settings:
        """Beanie 设置"""

        name = "event_log_records"

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
            # 用户ID和时间戳复合索引
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_user_timestamp",
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


class EventLogRecordProjection(DocumentBase, AuditBase):
    """
    事件日志简化模型（不包含向量）

    用于大部分不需要向量数据的场景，减少数据传输和内存占用。
    """

    # 核心字段
    id: Optional[PydanticObjectId] = Field(default=None, description="记录ID")
    user_id: Optional[str] = Field(default=None, description="用户ID，个人事件必填")
    user_name: Optional[str] = Field(default=None, description="用户名称")
    group_id: Optional[str] = Field(default=None, description="群组ID")
    group_name: Optional[str] = Field(default=None, description="群组名称")
    atomic_fact: str = Field(..., description="原子事实内容（单条句子）")
    parent_episode_id: str = Field(..., description="父情景记忆的 event_id")

    # 时间信息
    timestamp: datetime = Field(..., description="事件发生时间")

    # 群组和参与者信息
    participants: Optional[List[str]] = Field(default=None, description="相关参与者")

    # 向量模型信息（保留模型名称，但不包含向量数据）
    vector_model: Optional[str] = Field(default=None, description="使用的向量化模型")

    # 事件类型和扩展信息
    event_type: Optional[str] = Field(
        default=None, description="事件类型，如 Conversation"
    )
    extend: Optional[Dict[str, Any]] = Field(default=None, description="扩展字段")

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
            PydanticObjectId: lambda oid: str(oid),
        },
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """兼容性属性，返回文档ID"""
        return self.id


# 导出模型
__all__ = ["EventLogRecord", "EventLogRecordProjection"]
