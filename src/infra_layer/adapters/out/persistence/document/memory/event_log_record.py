"""
EventLogRecord Beanie ODM model

Unified storage for event logs (atomic facts) extracted from episodic memory (individual or group).
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase
from beanie import PydanticObjectId
from api_specs.memory_types import ParentType


class EventLogRecord(DocumentBase, AuditBase):
    """
    Generic event log document model

    Stores atomic facts split from individual or group episodic memory for fine-grained retrieval.
    """

    # Core fields
    user_id: Optional[str] = Field(
        default=None, description="User ID, required for personal events"
    )
    user_name: Optional[str] = Field(default=None, description="User name")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    group_name: Optional[str] = Field(default=None, description="Group name")
    atomic_fact: str = Field(..., description="Atomic fact content (single sentence)")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Time information
    timestamp: datetime = Field(..., description="Event occurrence time")

    # Group and participant information
    participants: Optional[List[str]] = Field(
        default=None, description="Related participants"
    )

    # Vector and model
    vector: Optional[List[float]] = Field(
        default=None, description="Atomic fact vector"
    )
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    # Event type and extension information
    event_type: Optional[str] = Field(
        default=None, description="Event type, such as Conversation"
    )
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Extension field"
    )

    model_config = ConfigDict(
        collection="event_log_records",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "id": "atomic_fact_001",
                "user_id": "user_12345",
                "user_name": "Alice",
                "atomic_fact": "The user went to Chengdu on January 1, 2024, and enjoyed the local Sichuan cuisine.",
                "parent_type": ParentType.MEMCELL.value,
                "parent_id": "memcell_001",
                "timestamp": "2024-01-01T10:00:00+00:00",
                "group_id": "group_travel",
                "group_name": "Travel Group",
                "participants": ["Zhang San", "Li Si"],
                "vector": [0.1, 0.2, 0.3],
                "event_type": "Conversation",
                "extend": {"location": "Chengdu"},
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id

    class Settings:
        """Beanie Settings"""

        name = "event_log_records"

        indexes = [
            # Single field indexes
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            IndexModel([("group_id", ASCENDING)], name="idx_group_id", sparse=True),
            IndexModel([("timestamp", DESCENDING)], name="idx_timestamp"),
            # Parent memory index
            IndexModel([("parent_id", ASCENDING)], name="idx_parent_id"),
            # Composite index of user ID and timestamp
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_user_timestamp",
            ),
            # Composite index of group ID and timestamp
            IndexModel(
                [("group_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_group_timestamp",
                sparse=True,
            ),
            # Composite index on group ID, user ID and timestamp
            # Note: This also covers (group_id, user_id) queries by left-prefix rule
            IndexModel(
                [
                    ("group_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_group_user_timestamp",
                sparse=True,
            ),
            # Creation time index
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            # Update time index
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]

        validate_on_save = True
        use_state_management = True


class EventLogRecordProjection(DocumentBase, AuditBase):
    """
    Simplified event log model (without vector)

    Used in most scenarios where vector data is not needed, reducing data transfer and memory usage.
    """

    # Core fields
    id: Optional[PydanticObjectId] = Field(default=None, description="Record ID")
    user_id: Optional[str] = Field(
        default=None, description="User ID, required for personal events"
    )
    user_name: Optional[str] = Field(default=None, description="User name")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    group_name: Optional[str] = Field(default=None, description="Group name")
    atomic_fact: str = Field(..., description="Atomic fact content (single sentence)")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Time information
    timestamp: datetime = Field(..., description="Event occurrence time")

    # Group and participant information
    participants: Optional[List[str]] = Field(
        default=None, description="Related participants"
    )

    # Vector model information (retain model name, but exclude vector data)
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    # Event type and extension information
    event_type: Optional[str] = Field(
        default=None, description="Event type, such as Conversation"
    )
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Extension field"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat(),
            PydanticObjectId: lambda oid: str(oid),
        },
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id


# Export models
__all__ = ["EventLogRecord", "EventLogRecordProjection"]
