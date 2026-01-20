"""
MemCell Beanie ODM model

MemCell data model definition based on Beanie ODM, supporting MongoDB sharded clusters.
"""

from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

from core.oxm.mongo.document_base_with_soft_delete import DocumentBaseWithSoftDelete
from pydantic import BaseModel, Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId
from core.oxm.mongo.audit_base import AuditBase


class DataTypeEnum(str, Enum):
    """Data type enumeration"""

    CONVERSATION = "Conversation"


class Message(BaseModel):
    """Message structure"""

    content: str = Field(..., description="Message text content")
    files: Optional[List[str]] = Field(default=None, description="List of file links")
    extend: Optional[Dict[str, str]] = Field(
        default=None, description="Extended fields"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Today's meeting discussed the design plan for the new feature",
                "files": ["https://example.com/design_doc.pdf"],
                "extend": {
                    "sender": "Zhang San",
                    "message_id": "msg_001",
                    "platform": "WeChat",
                },
            }
        }
    )


class RawData(BaseModel):
    """Raw data structure"""

    data_type: DataTypeEnum = Field(..., description="Data type enumeration")
    messages: List[Message] = Field(..., min_length=1, description="List of messages")
    meta: Optional[Dict[str, str]] = Field(default=None, description="Metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data_type": "Conversation",
                "messages": [
                    {
                        "content": "Team discussed new feature",
                        "extend": {"sender": "Zhang San"},
                    }
                ],
                "meta": {"chat_id": "chat_12345", "platform": "WeChat"},
            }
        }
    )


class MemCell(DocumentBaseWithSoftDelete, AuditBase):
    """
    MemCell document model

    Storage model for scene segmentation results, supporting flexible extension and high-performance queries.

    Supports soft delete functionality:
    - Use delete() method for soft deletion
    - Use find_one(), find_many() to automatically filter out deleted records
    - Use hard_find_one(), hard_find_many() to query including deleted records
    - Use hard_delete() for physical deletion
    """

    # Core fields (required)
    user_id: Optional[str] = Field(
        None,
        description="User ID, core query field. None for group memory, user ID for personal memory",
    )
    timestamp: datetime = Field(..., description="Occurrence time, shard key")
    summary: Optional[str] = Field(
        default=None,
        description="Memory unit summary, can be empty for force-split memcells",
    )

    # Optional fields
    group_id: Optional[str] = Field(
        default=None, description="Group ID, empty means private chat"
    )
    original_data: Optional[List] = Field(
        default=None, description="Original information"
    )
    participants: Optional[List[str]] = Field(
        default=None, description="Names of event participants"
    )
    type: Optional[DataTypeEnum] = Field(default=None, description="Scenario type")

    subject: Optional[str] = Field(default=None, description="Memory unit subject")

    keywords: Optional[List[str]] = Field(default=None, description="Keywords")
    linked_entities: Optional[List[str]] = Field(
        default=None, description="Associated entity IDs"
    )

    # Possibly unused
    episode: Optional[str] = Field(default=None, description="Scenario memory")
    foresight_memories: Optional[List] = Field(default=None, description="Foresight")
    event_log: Optional[Dict] = Field(
        default=None, description="Event Log atomic facts"
    )
    extend: Optional[Dict] = Field(default=None, description="Extended fields")

    model_config = ConfigDict(
        # Collection name
        collection="memcells",
        # Validation configuration
        validate_assignment=True,
        # JSON serialization configuration
        json_encoders={datetime: lambda dt: dt.isoformat()},
        # Example data
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "group_id": "group_67890",
                "timestamp": "2024-12-01T10:30:00.000Z",
                "summary": "Team discussed new feature design plan and received positive feedback",
                "original_data": [
                    {
                        "data_type": "Conversation",
                        "messages": [
                            {
                                "content": "Today's meeting discussed the design plan for the new feature",
                                "files": ["https://example.com/design_doc.pdf"],
                                "extend": {
                                    "sender": "Zhang San",
                                    "message_id": "msg_001",
                                },
                            }
                        ],
                        "meta": {"chat_id": "chat_12345", "platform": "WeChat"},
                    }
                ],
                "participants": ["Zhang San", "Li Si", "Wang Wu"],
                "type": "Conversation",
                "keywords": ["New feature", "Design plan", "Meeting"],
                "linked_entities": ["project_001", "feature_002"],
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """
        Beanie settings

        Note: MemCell is stored in KV-Storage only, not in MongoDB.
        No indexes are needed since this model is used for type definitions only.
        All indexes are defined in MemCellLite which is stored in MongoDB.
        """

        # Collection name (for compatibility, but not actually used in MongoDB)
        name = "memcells"

        # No indexes - MemCell is not stored in MongoDB, only in KV-Storage
        # All indexes are defined in MemCellLite
        indexes = []

        # Validation settings
        validate_on_save = True
        use_state_management = True


# Export models
__all__ = ["MemCell", "RawData", "Message", "DataTypeEnum"]
