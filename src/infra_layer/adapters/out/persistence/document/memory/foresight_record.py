"""
ForesightRecord Beanie ODM model

Unified storage of foresights extracted from episodic memories (personal or group).
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase
from beanie import PydanticObjectId
from api_specs.memory_types import ParentType


class ForesightRecord(DocumentBase, AuditBase):
    """
    Generic foresight document model

    Unified storage of foresight information extracted from personal or group episodic memories.
    When user_id exists, it represents personal foresight; when user_id is empty and group_id exists, it represents group foresight.
    """

    # Core fields
    user_id: Optional[str] = Field(
        default=None,
        description="User ID, required for personal memory, None for group memory",
    )
    user_name: Optional[str] = Field(default=None, description="User name")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    group_name: Optional[str] = Field(default=None, description="Group name")
    content: str = Field(..., min_length=1, description="Foresight content")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Time range fields
    start_time: Optional[str] = Field(
        default=None, description="Foresight start time (date string, e.g., 2024-01-01)"
    )
    end_time: Optional[str] = Field(
        default=None, description="Foresight end time (date string, e.g., 2024-12-31)"
    )
    duration_days: Optional[int] = Field(default=None, description="Duration in days")

    # Group and participant information
    participants: Optional[List[str]] = Field(
        default=None, description="Related participants"
    )

    # Vector and model
    vector: Optional[List[float]] = Field(
        default=None, description="Text vector of the foresight"
    )
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    # Evidence and extension information
    evidence: Optional[str] = Field(
        default=None, description="Evidence supporting this foresight"
    )
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Extension field"
    )

    model_config = ConfigDict(
        collection="foresight_records",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "id": "foresight_001",
                "user_id": "user_12345",
                "user_name": "Alice",
                "content": "User likes Sichuan cuisine, especially spicy hotpot",
                "parent_type": ParentType.MEMCELL.value,
                "parent_id": "memcell_001",
                "start_time": "2024-01-01",
                "end_time": "2024-12-31",
                "duration_days": 365,
                "group_id": "group_friends",
                "group_name": "Friends group",
                "participants": ["Zhang San", "Li Si"],
                "vector": [0.1, 0.2, 0.3],
                "vector_model": "text-embedding-3-small",
                "evidence": "Mentioned multiple times in chat about liking hotpot",
                "extend": {"confidence": 0.9},
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id

    class Settings:
        """Beanie settings"""

        name = "foresight_records"

        indexes = [
            # Single field indexes
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            IndexModel([("group_id", ASCENDING)], name="idx_group_id", sparse=True),
            # Parent memory index
            IndexModel([("parent_id", ASCENDING)], name="idx_parent_id"),
            # Composite index for time range queries (start_time, end_time)
            IndexModel(
                [("start_time", ASCENDING), ("end_time", ASCENDING)],
                name="idx_time_range",
                sparse=True,
            ),
            # Composite index of user ID and time range
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("start_time", ASCENDING),
                    ("end_time", ASCENDING),
                ],
                name="idx_user_time_range",
                sparse=True,
            ),
            # Composite index of group ID and time range
            IndexModel(
                [
                    ("group_id", ASCENDING),
                    ("start_time", ASCENDING),
                    ("end_time", ASCENDING),
                ],
                name="idx_group_time_range",
                sparse=True,
            ),
            # Composite index of group ID, user ID and time range
            # Note: This also covers (group_id, user_id) queries by left-prefix rule
            IndexModel(
                [
                    ("group_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("start_time", ASCENDING),
                    ("end_time", ASCENDING),
                ],
                name="idx_group_user_time_range",
                sparse=True,
            ),
            # Creation time index
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            # Update time index
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]

        validate_on_save = True
        use_state_management = True


class ForesightRecordProjection(DocumentBase, AuditBase):
    """
    Simplified foresight model (without vector)

    Used in most scenarios where vector data is not needed, reducing data transfer and memory usage.
    """

    # Core fields
    id: Optional[PydanticObjectId] = Field(default=None, description="Record ID")
    user_id: Optional[str] = Field(
        default=None,
        description="User ID, required for personal memory, None for group memory",
    )
    user_name: Optional[str] = Field(default=None, description="User name")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    group_name: Optional[str] = Field(default=None, description="Group name")
    content: str = Field(..., min_length=1, description="Foresight content")
    parent_type: str = Field(..., description="Parent memory type (memcell/episode)")
    parent_id: str = Field(..., description="Parent memory ID")

    # Time range fields
    start_time: Optional[str] = Field(
        default=None, description="Foresight start time (date string, e.g., 2024-01-01)"
    )
    end_time: Optional[str] = Field(
        default=None, description="Foresight end time (date string, e.g., 2024-12-31)"
    )
    duration_days: Optional[int] = Field(default=None, description="Duration in days")

    # Group and participant information
    participants: Optional[List[str]] = Field(
        default=None, description="Related participants"
    )

    # Vector model information (retain model name, but exclude vector data)
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    # Evidence and extension information
    evidence: Optional[str] = Field(
        default=None, description="Evidence supporting this foresight"
    )
    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Extension field"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat(), PydanticObjectId: str},
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        """Compatibility property, returns document ID"""
        return self.id


# Export models
__all__ = ["ForesightRecord", "ForesightRecordProjection"]
