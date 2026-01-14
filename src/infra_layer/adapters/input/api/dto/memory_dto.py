# -*- coding: utf-8 -*-
"""
Memory API DTO

Request and response data transfer objects for Memory API.
These models are used to define OpenAPI parameter documentation.
"""

from token import OP
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from core.oxm.constants import MAGIC_ALL
from api_specs.memory_models import MessageSenderRole


class MemorizeMessageRequest(BaseModel):
    """
    Store single message request body

    Used for POST /api/v1/memories endpoint
    """

    group_id: Optional[str] = Field(
        default=None, description="Group ID", examples=["group_123"]
    )
    group_name: Optional[str] = Field(
        default=None, description="Group name", examples=["Project Discussion Group"]
    )
    message_id: str = Field(
        ..., description="Message unique identifier", examples=["msg_001"]
    )
    create_time: str = Field(
        ...,
        description="Message creation time (ISO 8601 format)",
        examples=["2025-01-15T10:00:00+00:00"],
    )
    sender: str = Field(..., description="Sender user ID", examples=["user_001"])
    sender_name: Optional[str] = Field(
        default=None,
        description="Sender name (uses sender if not provided)",
        examples=["John"],
    )
    role: Optional[str] = Field(
        default=None,
        description="""Message sender role, used to identify the source of the message.
Enum values from MessageSenderRole:
- user: Message from a human user
- assistant: Message from an AI assistant""",
        examples=["user", "assistant"],
    )
    content: str = Field(
        ...,
        description="Message content",
        examples=["Let's discuss the technical solution for the new feature today"],
    )
    refer_list: Optional[List[str]] = Field(
        default=None,
        description="List of referenced message IDs",
        examples=[["msg_000"]],
    )

    @model_validator(mode="after")
    def validate_role(self):
        """Validate that role is a valid MessageSenderRole value"""
        if self.role is not None and not MessageSenderRole.is_valid(self.role):
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of: {[r.value for r in MessageSenderRole]}"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "group_id": "group_123",
                "group_name": "Project Discussion Group",
                "message_id": "msg_001",
                "create_time": "2025-01-15T10:00:00+00:00",
                "sender": "user_001",
                "sender_name": "John",
                "role": "user",
                "content": "Let's discuss the technical solution for the new feature today",
                "refer_list": ["msg_000"],
            }
        }
    }


class FetchMemoriesParams(BaseModel):
    """
    Query parameters for fetching user memories

    Used for GET /api/v1/memories endpoint
    """

    user_id: str = Field(..., description="User ID", examples=["user_123"])
    memory_type: Optional[str] = Field(
        default="episodic_memory",
        description="""Memory type, enum values from MemoryType:
- profile: user profile
- episodic_memory: episodic memory (default)
- foresight: prospective memory
- event_log: event log (atomic facts)""",
        examples=["profile"],
    )
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of memories to return",
        ge=1,
        le=100,
        examples=[20],
    )
    offset: Optional[int] = Field(
        default=0, description="Pagination offset", ge=0, examples=[0]
    )
    sort_by: Optional[str] = Field(
        default=None, description="Sort field", examples=["created_at"]
    )
    sort_order: Optional[str] = Field(
        default="desc",
        description="""Sort direction, enum values:
- asc: ascending order
- desc: descending order (default)""",
        examples=["desc"],
    )
    version_range: Optional[List[Optional[str]]] = Field(
        default=None,
        description="Version range filter, format [start, end], closed interval",
        examples=[["v1.0", "v2.0"]],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "memory_type": "profile",
                "limit": 20,
                "offset": 0,
                "sort_order": "desc",
            }
        }
    }


class SearchMemoriesRequest(BaseModel):
    """
    Search memories request parameters

    Used for GET /api/v1/memories/search endpoint
    Supports passing parameters via query params or body
    """

    user_id: Optional[str] = Field(
        default=None,
        description="User ID (at least one of user_id or group_id must be provided)",
        examples=["user_123"],
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Group ID (at least one of user_id or group_id must be provided)",
        examples=["group_456"],
    )
    query: Optional[str] = Field(
        default=None, description="Search query text", examples=["coffee preference"]
    )
    retrieve_method: Optional[str] = Field(
        default="keyword",
        description="""Retrieval method, enum values from RetrieveMethod:
- keyword: keyword retrieval (BM25, default)
- vector: vector semantic retrieval
- hybrid: hybrid retrieval (keyword + vector)
- rrf: RRF fusion retrieval (keyword + vector + RRF ranking fusion)
- agentic: LLM-guided multi-round intelligent retrieval""",
        examples=["keyword"],
    )
    top_k: Optional[int] = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=100,
        examples=[10],
    )
    memory_types: Optional[List[str]] = Field(
        default=None,
        description="""List of memory types to retrieve, enum values from MemoryType:
- episodic_memory: episodic memory
- foresight: prospective memory
- event_log: event log (atomic facts)
Note: profile type is not supported in search interface""",
        examples=[["episodic_memory"]],
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Time range start (ISO 8601 format)",
        examples=["2024-01-01T00:00:00"],
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Time range end (ISO 8601 format)",
        examples=["2024-12-31T23:59:59"],
    )
    radius: Optional[float] = Field(
        default=None,
        description="COSINE similarity threshold for vector retrieval (only for vector and hybrid methods, default 0.6)",
        ge=0.0,
        le=1.0,
        examples=[0.6],
    )
    include_metadata: Optional[bool] = Field(
        default=True, description="Whether to include metadata", examples=[True]
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional filter conditions", examples=[{}]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "query": "coffee preference",
                "retrieve_method": "keyword",
                "top_k": 10,
                "memory_types": ["episodic_memory"],
            }
        }
    }


class UserDetailRequest(BaseModel):
    """User detail request model"""

    full_name: Optional[str] = Field(
        default=None, description="User full name", examples=["John Smith"]
    )
    role: Optional[str] = Field(
        default=None,
        description="""User type role, used to identify if this user is a human or AI.
Enum values from MessageSenderRole:
- user: Human user
- assistant: AI assistant/bot""",
        examples=["user", "assistant"],
    )
    custom_role: Optional[str] = Field(
        default=None,
        description="User's job/position role (e.g. developer, designer, manager)",
        examples=["developer"],
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional information",
        examples=[{"department": "Engineering"}],
    )

    @model_validator(mode="after")
    def validate_role(self):
        """Validate that role is a valid MessageSenderRole value"""
        if self.role is not None and not MessageSenderRole.is_valid(self.role):
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of: {[r.value for r in MessageSenderRole]}"
            )
        return self


class ConversationMetaCreateRequest(BaseModel):
    """
    Save conversation metadata request body

    Used for POST /api/v1/memories/conversation-meta endpoint
    """

    version: str = Field(..., description="Metadata version number", examples=["1.0"])
    scene: str = Field(
        ...,
        description="""Scene identifier, enum values from ScenarioType:
- group_chat: work/group chat scenario, suitable for group conversations such as multi-person collaboration and project discussions
- assistant: assistant scenario, suitable for one-on-one AI assistant conversations""",
        examples=["group_chat"],
    )
    scene_desc: Dict[str, Any] = Field(
        ...,
        description="Scene description object, can include fields like bot_ids",
        examples=[{"bot_ids": ["bot_001"], "type": "project_discussion"}],
    )
    name: str = Field(
        ..., description="Conversation name", examples=["Project Discussion Group"]
    )
    description: Optional[str] = Field(
        default=None,
        description="Conversation description",
        examples=["Technical discussion for new feature development"],
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Group unique identifier. When null/not provided, represents default settings for this scene.",
        examples=["group_123", None],
    )
    created_at: str = Field(
        ...,
        description="Conversation creation time (ISO 8601 format)",
        examples=["2025-01-15T10:00:00+00:00"],
    )
    default_timezone: Optional[str] = Field(
        default=None, description="Default timezone", examples=["UTC"]
    )
    user_details: Optional[Dict[str, UserDetailRequest]] = Field(
        default=None,
        description="Participant details, key is user ID, value is user detail object",
        examples=[
            {
                "user_001": {
                    "full_name": "John Smith",
                    "role": "user",
                    "custom_role": "developer",
                    "extra": {"department": "Engineering"},
                },
                "bot_001": {
                    "full_name": "AI Assistant",
                    "role": "assistant",
                    "custom_role": "assistant",
                    "extra": {"type": "ai"},
                },
            }
        ],
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Tag list", examples=[["work", "technical"]]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "With group_id",
                    "value": {
                        "version": "1.0",
                        "scene": "group_chat",
                        "scene_desc": {
                            "bot_ids": ["bot_001"],
                            "type": "project_discussion",
                        },
                        "name": "Project Discussion Group",
                        "description": "Technical discussion for new feature development",
                        "group_id": "group_123",
                        "created_at": "2025-01-15T10:00:00+00:00",
                        "default_timezone": "UTC",
                        "user_details": {
                            "user_001": {
                                "full_name": "John Smith",
                                "role": "user",
                                "custom_role": "developer",
                                "extra": {"department": "Engineering"},
                            },
                            "bot_001": {
                                "full_name": "AI Assistant",
                                "role": "assistant",
                            },
                        },
                        "tags": ["work", "technical"],
                    },
                },
                {
                    "summary": "Default config (group_id is null)",
                    "value": {
                        "version": "1.0",
                        "scene": "group_chat",
                        "scene_desc": {"bot_ids": ["default_bot"]},
                        "name": "Default Group Chat Settings",
                        "description": "Default settings for group_chat scene",
                        "group_id": None,
                        "created_at": "2025-01-15T10:00:00+00:00",
                        "default_timezone": "UTC",
                        "tags": ["default"],
                    },
                },
            ]
        }
    }


class ConversationMetaGetRequest(BaseModel):
    """
    Get conversation metadata request parameters

    Used for GET /api/v1/memories/conversation-meta endpoint
    """

    group_id: Optional[str] = Field(
        default=None,
        description="Group ID to look up. If not found, will automatically fallback to default config (group_id=null). If not provided, returns default config directly.",
        examples=["group_123"],
    )

    model_config = {"json_schema_extra": {"example": {"group_id": "group_123"}}}


class ConversationMetaPatchRequest(BaseModel):
    """
    Partial update conversation metadata request body

    Used for PATCH /api/v1/memories/conversation-meta endpoint
    """

    group_id: Optional[str] = Field(
        default=None,
        description="Group ID to update. When null, updates the default config.",
        examples=["group_123", None],
    )
    name: Optional[str] = Field(
        default=None,
        description="New conversation name",
        examples=["New Conversation Name"],
    )
    description: Optional[str] = Field(
        default=None,
        description="New conversation description",
        examples=["Updated description"],
    )
    scene_desc: Optional[Dict[str, Any]] = Field(
        default=None,
        description="New scene description",
        examples=[{"bot_ids": ["bot_002"]}],
    )
    tags: Optional[List[str]] = Field(
        default=None, description="New tag list", examples=[["tag1", "tag2"]]
    )
    user_details: Optional[Dict[str, UserDetailRequest]] = Field(
        default=None,
        description="New user details (will completely replace existing user_details)",
        examples=[
            {
                "user_001": {
                    "full_name": "John Smith",
                    "role": "user",
                    "custom_role": "lead",
                }
            }
        ],
    )
    default_timezone: Optional[str] = Field(
        default=None, description="New default timezone", examples=["Asia/Shanghai"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Update by group_id",
                    "value": {
                        "group_id": "group_123",
                        "name": "New Conversation Name",
                        "tags": ["updated", "tags"],
                    },
                },
                {
                    "summary": "Update default config (group_id is null)",
                    "value": {"group_id": None, "name": "Updated Default Settings"},
                },
            ]
        }
    }


class DeleteMemoriesRequest(BaseModel):
    """
    Delete memories request body

    Used for DELETE /api/v1/memories endpoint

    Notes:
    - event_id, user_id, group_id are combined filter conditions
    - If all three are provided, all conditions must be met
    - If not provided, use MAGIC_ALL ("__all__") to skip filtering
    - Cannot all be MAGIC_ALL (at least one filter required)
    """

    event_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="Memory event_id (filter condition)",
        examples=["507f1f77bcf86cd799439011", MAGIC_ALL],
    )
    user_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="User ID (filter condition)",
        examples=["user_123", MAGIC_ALL],
    )
    group_id: Optional[str] = Field(
        default=MAGIC_ALL,
        description="Group ID (filter condition)",
        examples=["group_456", MAGIC_ALL],
    )

    @model_validator(mode="after")
    def validate_filters(self):
        """Validate that at least one filter is provided"""
        # Check if all are MAGIC_ALL
        if (
            self.event_id == MAGIC_ALL
            and self.user_id == MAGIC_ALL
            and self.group_id == MAGIC_ALL
        ):
            raise ValueError(
                "At least one of event_id, user_id, or group_id must be provided (not MAGIC_ALL)"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Delete by event_id only",
                    "value": {
                        "event_id": "507f1f77bcf86cd799439011",
                        "user_id": MAGIC_ALL,
                        "group_id": MAGIC_ALL,
                    },
                },
                {
                    "summary": "Delete by user_id only",
                    "value": {
                        "event_id": MAGIC_ALL,
                        "user_id": "user_123",
                        "group_id": MAGIC_ALL,
                    },
                },
                {
                    "summary": "Delete by user_id and group_id",
                    "value": {
                        "event_id": MAGIC_ALL,
                        "user_id": "user_123",
                        "group_id": "group_456",
                    },
                },
            ]
        }
    }


class ConversationMetaResponse(BaseModel):
    """
    Conversation metadata response DTO

    Used for GET /api/v1/memories/conversation-meta response
    """

    id: str = Field(..., description="Document ID")
    group_id: Optional[str] = Field(
        default=None, description="Group ID (null for default config)"
    )
    scene: str = Field(..., description="Scene identifier")
    scene_desc: Optional[Dict[str, Any]] = Field(
        default=None, description="Scene description"
    )
    name: str = Field(..., description="Conversation name")
    description: Optional[str] = Field(
        default=None, description="Conversation description"
    )
    version: str = Field(..., description="Metadata version")
    conversation_created_at: str = Field(..., description="Conversation creation time")
    default_timezone: Optional[str] = Field(
        default=None, description="Default timezone"
    )
    user_details: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="User details"
    )
    tags: List[str] = Field(default_factory=list, description="Tags")
    is_default: bool = Field(
        default=False, description="Whether this is the default config"
    )
    created_at: Optional[str] = Field(default=None, description="Record creation time")
    updated_at: Optional[str] = Field(default=None, description="Record update time")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "group_id": "group_123",
                "scene": "group_chat",
                "scene_desc": {"bot_ids": ["bot_001"]},
                "name": "Project Discussion",
                "description": "Technical discussion group",
                "version": "1.0",
                "conversation_created_at": "2025-01-15T10:00:00+00:00",
                "default_timezone": "UTC",
                "user_details": {
                    "user_001": {
                        "full_name": "John",
                        "role": "user",
                        "custom_role": "developer",
                    },
                    "bot_001": {"full_name": "AI Assistant", "role": "assistant"},
                },
                "tags": ["work", "tech"],
                "is_default": False,
                "created_at": "2025-01-15T10:00:00+00:00",
                "updated_at": "2025-01-15T10:00:00+00:00",
            }
        }
    }
