"""
ConversationMeta Beanie ODM 模型

基于 Beanie ODM 的对话元数据文档模型，存储对话的完整元信息。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict, BaseModel
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from core.oxm.mongo.audit_base import AuditBase


class UserDetailModel(BaseModel):
    """用户详情嵌套模型
    
    用于存储用户的基本信息和额外扩展信息
    """

    full_name: str = Field(..., description="用户全名")
    role: Optional[str] = Field(default=None, description="用户角色，如：用户、助手、管理员等")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="扩展字段，支持动态schema")


class ConversationMeta(DocumentBase, AuditBase):
    """
    对话元数据文档模型

    存储对话的完整元信息，包括场景、参与者、标签等。
    用于多轮对话的上下文管理和记忆提取。
    """

    # 版本信息
    version: str = Field(..., description="数据版本号，如：1.0.0")

    # 场景信息
    scene: str = Field(..., description="场景标识符，用于区分不同的应用场景")
    scene_desc: Dict[str, Any] = Field(
        default_factory=dict, description="场景描述信息，通常包含bot_ids等字段"
    )

    # 对话基本信息
    name: str = Field(..., description="对话名称")
    description: Optional[str] = Field(default=None, description="对话描述")
    group_id: Indexed(str) = Field(..., description="群组ID，用于关联同一组对话")

    # 时间信息
    conversation_created_at: str = Field(..., description="对话创建时间，ISO格式字符串")
    default_timezone: Optional[str] = Field(
        default="Asia/Shanghai", description="默认时区，如：Asia/Shanghai"
    )

    # 参与者信息
    user_details: Dict[str, UserDetailModel] = Field(
        default_factory=dict,
        description="参与者详情字典，key为动态用户ID（如user_001, robot_001），value为用户详情",
    )

    # 标签和分类
    tags: List[str] = Field(default_factory=list, description="标签列表，用于分类和检索")

    model_config = ConfigDict(
        # 集合名称
        collection="conversation_metas",
        # 验证配置
        validate_assignment=True,
        # JSON 序列化配置
        json_encoders={datetime: lambda dt: dt.isoformat()},
        # 示例数据
        json_schema_extra={
            "example": {
                "version": "1.0.0",
                "scene": "scene_a",
                "scene_desc": {"bot_ids": ["aaa", "bbb", "ccc"]},
                "name": "用户健康咨询对话",
                "description": "用户与AI助手关于北京旅游、健康管理、运动康复等主题的对话记录",
                "group_id": "chat_user_001_assistant",
                "conversation_created_at": "2025-08-26T08:00:00+08:00",
                "default_timezone": "Asia/Shanghai",
                "user_details": {
                    "user_001": {
                        "full_name": "用户",
                        "role": "用户",
                        "extra": {
                            "height": 170,
                            "weight": 86,
                            "bmi": 29.8,
                            "waist_circumference": 104,
                            "origin": "四川",
                            "preferences": {"food": "火锅", "activities": "团体活动"},
                        },
                    },
                    "robot_001": {
                        "full_name": "AI助手",
                        "role": "助手",
                        "extra": {"type": "assistant"},
                    },
                },
                "tags": ["健康咨询", "旅游规划", "运动康复", "饮食建议"],
            }
        },
    )

    class Settings:
        """Beanie 设置"""

        name = "conversation_metas"
        indexes = [
            # group_id 索引（高频查询）
            IndexModel([("group_id", ASCENDING)], name="idx_group_id"),
            # scene 索引（场景查询）
            IndexModel([("scene", ASCENDING)], name="idx_scene"),
            # 组合索引：group_id + scene（常见的复合查询）
            IndexModel(
                [("group_id", ASCENDING), ("scene", ASCENDING)],
                name="idx_group_id_scene",
            ),
        ]
        validate_on_save = True
        use_state_management = True

