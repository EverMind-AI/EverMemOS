"""
记忆数据模型定义

此模块包含 fetch_mem_service 的输入输出数据结构定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class RetrieveMethod(str, Enum):
    """检索方法枚举"""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class MemoryType(str, Enum):
    """记忆类型枚举"""

    BASE_MEMORY = "base_memory"
    PROFILE = "profile"
    PREFERENCE = "preference"

    # core就是multiple，就是上面三个
    MULTIPLE = "multiple"  # 多类型查询
    CORE = "core"  # 核心记忆

    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"  # 暂时先不删除等着xinze确认
    ENTITY = "entity"
    RELATION = "relation"
    BEHAVIOR_HISTORY = "behavior_history"

    PERSONAL_SEMANTIC_MEMORY = "personal_semantic_memory"
    PERSONAL_EVENT_LOG = "personal_event_log"  # 事件日志（原子事实）

    GROUP_PROFILE = "group_profile"  # 群组画像


@dataclass
class Metadata:
    """记忆元数据类"""

    # 必需字段
    source: str  # 数据来源
    user_id: str  # 用户ID
    memory_type: str  # 记忆类型

    # 可选字段
    limit: Optional[int] = None  # 限制数量
    email: Optional[str] = None  # 邮箱
    phone: Optional[str] = None  # 电话
    full_name: Optional[str] = None  # 全名

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metadata':
        """从字典创建 Metadata 对象"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class BaseMemoryModel:
    """基础记忆模型"""

    id: str
    user_id: str
    content: str
    created_at: datetime
    updated_at: datetime
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class ProfileModel:
    """用户画像模型"""

    id: str
    user_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class PreferenceModel:
    """用户偏好模型"""

    id: str
    user_id: str
    category: str
    preference_key: str
    preference_value: Any
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class EpisodicMemoryModel:
    """情景记忆模型"""

    id: str
    user_id: str
    episode_id: str  # 就是id，没区别，为了兼容性先留着
    title: str
    summary: str
    timestamp: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    key_events: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)
    extend: Optional[Dict[str, Any]] = None
    memcell_event_id_list: Optional[List[str]] = None
    subject: Optional[str] = None


@dataclass
class SemanticMemoryModel:
    """语义记忆模型"""

    id: str
    user_id: str
    concept: str
    definition: str
    category: str
    related_concepts: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class EntityModel:
    """实体模型"""

    id: str
    user_id: str
    entity_name: str
    entity_type: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class RelationModel:
    """关系模型"""

    id: str
    user_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    relation_description: str
    strength: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class BehaviorHistoryModel:
    """行为历史模型"""

    id: str
    user_id: str
    action_type: str
    action_description: str
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class CoreMemoryModel:
    """核心记忆模型"""

    id: str
    user_id: str
    version: str
    is_latest: bool

    # ==================== BaseMemory 字段 ====================
    user_name: Optional[str] = None
    gender: Optional[str] = None
    position: Optional[str] = None
    supervisor_user_id: Optional[str] = None
    team_members: Optional[List[str]] = None
    okr: Optional[List[Dict[str, str]]] = None
    base_location: Optional[str] = None
    hiredate: Optional[str] = None
    age: Optional[int] = None
    department: Optional[str] = None

    # ==================== Profile 字段 ====================
    hard_skills: Optional[List[Dict[str, str]]] = None
    soft_skills: Optional[List[Dict[str, str]]] = None
    output_reasoning: Optional[str] = None
    motivation_system: Optional[List[Dict[str, Any]]] = None
    fear_system: Optional[List[Dict[str, Any]]] = None
    value_system: Optional[List[Dict[str, Any]]] = None
    humor_use: Optional[List[Dict[str, Any]]] = None
    colloquialism: Optional[List[Dict[str, Any]]] = None
    personality: Optional[Union[List[str], str]] = None
    way_of_decision_making: Optional[List[Dict[str, Any]]] = None
    projects_participated: Optional[List[Dict[str, str]]] = None
    user_goal: Optional[List[str]] = None
    work_responsibility: Optional[str] = None
    working_habit_preference: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    tendency: Optional[List[str]] = None

    # ==================== 通用字段 ====================
    extend: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class EventLogModel:
    """事件日志模型（原子事实）

    从情景记忆中提取的原子事实，用于细粒度检索。
    """

    id: str
    user_id: str
    atomic_fact: str  # 原子事实内容
    parent_episode_id: str  # 父情景记忆ID
    timestamp: datetime  # 事件发生时间

    # 可选字段
    user_name: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    participants: Optional[List[str]] = None
    vector: Optional[List[float]] = None
    vector_model: Optional[str] = None
    event_type: Optional[str] = None
    extend: Optional[Dict[str, Any]] = None

    # 通用时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class SemanticMemoryRecordModel:
    """语义记忆记录模型

    从情景记忆中提取的语义信息，支持个人和群组语义记忆。
    """

    id: str
    content: str  # 语义记忆内容
    parent_episode_id: str  # 父情景记忆ID

    # 可选字段
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    start_time: Optional[str] = None  # 开始时间（日期字符串）
    end_time: Optional[str] = None  # 结束时间（日期字符串）
    duration_days: Optional[int] = None  # 持续天数
    participants: Optional[List[str]] = None
    vector: Optional[List[float]] = None
    vector_model: Optional[str] = None
    evidence: Optional[str] = None  # 支持该语义记忆的证据
    extend: Optional[Dict[str, Any]] = None

    # 通用时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Metadata = field(default_factory=Metadata)


# 联合类型定义
MemoryModel = Union[
    BaseMemoryModel,
    ProfileModel,
    PreferenceModel,
    EpisodicMemoryModel,
    SemanticMemoryModel,
    EntityModel,
    RelationModel,
    BehaviorHistoryModel,
    CoreMemoryModel,
    EventLogModel,
    SemanticMemoryRecordModel,
]
