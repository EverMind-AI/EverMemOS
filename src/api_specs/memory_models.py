"""
Memory data model definitions

This module contains input and output data structure definitions for fetch_mem_service
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from common_utils.datetime_utils import get_now_with_timezone


class MessageSenderRole(str, Enum):
    """Enumeration of message sender roles

    Used to identify the source of a message in conversations.
    Compatible with OpenAI/mem0/memos message format.

    Values:
        USER: Message from a human user
        ASSISTANT: Message from an AI assistant
    """

    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def from_string(cls, role_str: Optional[str]) -> Optional['MessageSenderRole']:
        """
        Convert string to MessageSenderRole enum

        Args:
            role_str: Role string, such as "user", "assistant"

        Returns:
            MessageSenderRole enum value, returns None if conversion fails
        """
        if not role_str:
            return None

        try:
            role_lower = role_str.lower()
            for role in cls:
                if role.value == role_lower:
                    return role
            return None
        except Exception:
            return None

    @classmethod
    def is_valid(cls, role_str: Optional[str]) -> bool:
        """
        Check if the given string is a valid role

        Args:
            role_str: Role string to validate

        Returns:
            True if valid, False otherwise
        """
        if not role_str:
            return True  # None is allowed (optional field)
        return cls.from_string(role_str) is not None


class RetrieveMethod(str, Enum):
    """Enumeration of retrieval methods"""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    RRF = "rrf"  # keyword + vector + RRF fusion
    AGENTIC = "agentic"  # LLM-guided multi-round retrieval


class MemoryType(str, Enum):
    """Enumeration of memory types

    Currently implemented types:
    - PROFILE: User profile
    - EPISODIC_MEMORY: Episodic memory
    - FORESIGHT: Prospective memory
    - EVENT_LOG: Event log (atomic facts)

    Not yet implemented or deprecated:
    - BASE_MEMORY, PREFERENCE, CORE, ENTITY, RELATION, BEHAVIOR_HISTORY, GROUP_PROFILE
    """

    # ===== Implemented =====
    PROFILE = "profile"  # User profile
    EPISODIC_MEMORY = "episodic_memory"  # Episodic memory
    FORESIGHT = "foresight"  # Prospective memory
    EVENT_LOG = "event_log"  # Event log (atomic facts)

    # ===== Not yet implemented or deprecated =====
    BASE_MEMORY = "base_memory"  # [Not implemented]
    PREFERENCE = "preference"  # [Not implemented]
    CORE = "core"  # [Not implemented] Core memory
    ENTITY = "entity"  # [Not implemented]
    RELATION = "relation"  # [Not implemented]
    BEHAVIOR_HISTORY = "behavior_history"  # [Not implemented]
    GROUP_PROFILE = "group_profile"  # [Not implemented] Group profile


@dataclass
class Metadata:
    """Memory metadata class"""

    # Required fields
    source: str  # Data source
    user_id: str  # User ID
    memory_type: str  # Memory type

    # Optional fields
    group_id: Optional[str] = None  # Group ID
    limit: Optional[int] = None  # Limit count
    email: Optional[str] = None  # Email
    phone: Optional[str] = None  # Phone number
    full_name: Optional[str] = None  # Full name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metadata':
        """Create Metadata object from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class BaseMemoryModel:
    """Base memory model"""

    id: str
    user_id: str
    content: str
    created_at: datetime
    updated_at: datetime
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class ProfileModel:
    """User profile model"""

    id: str
    user_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class PreferenceModel:
    """User preference model"""

    id: str
    user_id: str
    category: str
    preference_key: str
    preference_value: Any
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class EpisodicMemoryModel:
    """Episodic memory model"""

    id: str
    user_id: str
    episode_id: str  # Same as id, no difference, kept for compatibility
    title: str
    summary: str
    timestamp: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    start_time: datetime = field(default_factory=get_now_with_timezone)
    end_time: Optional[datetime] = None
    key_events: List[str] = field(default_factory=list)
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)
    extend: Optional[Dict[str, Any]] = None
    memcell_event_id_list: Optional[List[str]] = None
    subject: Optional[str] = None


@dataclass
class ForesightModel:
    """Prospective memory model"""

    id: str
    user_id: str
    concept: str
    definition: str
    category: str
    related_concepts: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    source: Optional[str] = None
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class EntityModel:
    """Entity model"""

    id: str
    user_id: str
    entity_name: str
    entity_type: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class RelationModel:
    """Relation model"""

    id: str
    user_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    relation_description: str
    strength: float = 1.0
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class BehaviorHistoryModel:
    """Behavior history model"""

    id: str
    user_id: str
    action_type: str
    action_description: str
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    timestamp: datetime = field(default_factory=get_now_with_timezone)
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class CoreMemoryModel:
    """Core memory model"""

    id: str
    user_id: str
    version: str
    is_latest: bool

    # ==================== BaseMemory fields ====================
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

    # ==================== Profile fields ====================
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

    # ==================== Common fields ====================
    extend: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class EventLogModel:
    """Event log model (atomic facts)

    Atomic facts extracted from episodic memories, used for fine-grained retrieval.
    """

    id: str
    user_id: str
    atomic_fact: str  # Content of the atomic fact
    parent_type: str  # Parent memory type (memcell/episode)
    parent_id: str  # Parent memory ID
    timestamp: datetime  # Event occurrence time

    # Optional fields
    user_name: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    participants: Optional[List[str]] = None
    vector: Optional[List[float]] = None
    vector_model: Optional[str] = None
    event_type: Optional[str] = None
    extend: Optional[Dict[str, Any]] = None

    # Common timestamps
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class ForesightRecordModel:
    """Prospective record model

    Prospective information extracted from episodic memories, supporting individual and group foresight.
    """

    id: str
    content: str  # Prospective content
    parent_type: str  # Parent memory type (memcell/episode)
    parent_id: str  # Parent memory ID

    # Optional fields
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    start_time: Optional[str] = None  # Start time (date string)
    end_time: Optional[str] = None  # End time (date string)
    duration_days: Optional[int] = None  # Duration in days
    participants: Optional[List[str]] = None
    vector: Optional[List[float]] = None
    vector_model: Optional[str] = None
    evidence: Optional[str] = None  # Evidence supporting this foresight
    extend: Optional[Dict[str, Any]] = None

    # Common timestamps
    created_at: datetime = field(default_factory=get_now_with_timezone)
    updated_at: datetime = field(default_factory=get_now_with_timezone)
    metadata: Metadata = field(default_factory=Metadata)


# Union type definition
MemoryModel = Union[
    BaseMemoryModel,
    ProfileModel,
    PreferenceModel,
    EpisodicMemoryModel,
    ForesightModel,
    EntityModel,
    RelationModel,
    BehaviorHistoryModel,
    CoreMemoryModel,
    EventLogModel,
    ForesightRecordModel,
]
