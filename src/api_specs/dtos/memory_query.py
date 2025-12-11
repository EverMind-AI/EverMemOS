from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from api_specs.memory_types import BaseMemory
from api_specs.memory_models import MemoryType, Metadata, MemoryModel, RetrieveMethod
from common_utils.datetime_utils import get_timezone


@dataclass
class FetchMemRequest:
    """Memory retrieval request"""

    user_id: str
    limit: Optional[int] = 40
    offset: Optional[int] = 0
    filters: Optional[Dict[str, Any]] = field(default_factory=dict)
    memory_type: Optional[MemoryType] = MemoryType.MULTIPLE
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # "asc" or "desc"
    version_range: Optional[tuple[Optional[str], Optional[str]]] = (
        None  # Version range (start, end), closed interval [start, end]
    )

    def get_memory_types(self) -> List[MemoryType]:
        """Get the list of memory types to query"""
        if self.memory_type == MemoryType.MULTIPLE:
            # When MULTIPLE, return BASE_MEMORY, PROFILE, PREFERENCE
            return [MemoryType.BASE_MEMORY, MemoryType.PROFILE, MemoryType.PREFERENCE]
        else:
            return [self.memory_type]


@dataclass
class FetchMemResponse:
    """Memory retrieval response"""

    memories: List[MemoryModel]
    total_count: int
    has_more: bool = False
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class RetrieveMemRequest:
    """Memory retrieval request"""

    user_id: Optional[str] = None
    group_id: Optional[str] = None  # Group ID for group memory retrieval
    memory_types: List[MemoryType] = field(default_factory=list)
    top_k: int = 40
    filters: Dict[str, Any] = field(default_factory=dict)
    include_metadata: bool = True
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    query: Optional[str] = None  # when retrieving
    retrieve_method: RetrieveMethod = field(default=RetrieveMethod.KEYWORD)
    current_time: Optional[str] = (
        None  # Current time, used to filter forward-looking events within validity period happened_at current_time
    )
    radius: Optional[float] = (
        None  # COSINE similarity threshold (use default 0.6 if None)
    )


@dataclass
class RetrieveMemResponse:
    """Memory retrieval response"""

    memories: List[Dict[str, List[BaseMemory]]]
    scores: List[Dict[str, List[float]]]
    importance_scores: List[float] = field(
        default_factory=list
    )  # New: group importance scores
    original_data: List[Dict[str, List[Dict[str, Any]]]] = field(
        default_factory=list
    )  # New: original data
    total_count: int = 0
    has_more: bool = False
    query_metadata: Metadata = field(default_factory=Metadata)
    metadata: Metadata = field(default_factory=Metadata)


@dataclass
class UserDetail:
    """User details

    Structure for the value of ConversationMetaRequest.user_details
    """

    full_name: str  # User's full name
    role: Optional[str] = None  # User role
    extra: Optional[Dict[str, Any]] = None  # Additional information, schema is dynamic


@dataclass
class ConversationMetaRequest:
    """Conversation metadata request"""

    version: str  # Version number
    scene: str  # Scene identifier
    scene_desc: Dict[
        str, Any
    ]  # Scene description, usually contains fields like bot_ids
    name: str  # Conversation name
    group_id: str  # Group ID
    created_at: str  # Creation time, ISO format string
    description: Optional[str] = None  # Conversation description
    default_timezone: Optional[str] = get_timezone().key  # Default timezone
    user_details: Dict[str, UserDetail] = field(
        default_factory=dict
    )  # User details, key is dynamic (e.g., user_001, robot_001), value structure is fixed
    tags: List[str] = field(default_factory=list)  # List of tags
