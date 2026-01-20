"""
ForesightRecord Pydantic model

Unified storage of foresights extracted from episodic memories (personal or group).
Note: This model is stored in KV-Storage only (not MongoDB).
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from typing import Optional as OptionalType
from pydantic import BaseModel, Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId
from api_specs.memory_types import ParentType


class ForesightRecord(BaseModel):
    """
    Generic foresight document model

    Unified storage of foresight information extracted from personal or group episodic memories.
    When user_id exists, it represents personal foresight; when user_id is empty and group_id exists, it represents group foresight.

    Note: This model is stored in KV-Storage only. MongoDB stores ForesightRecordLite with indexed fields.
    """

    # ID field (managed by MongoDB through ForesightRecordLite)
    id: OptionalType[PydanticObjectId] = Field(
        default=None, description="Document ID (set after MongoDB insert)"
    )

    # Audit fields (managed by MongoDB through ForesightRecordLite)
    created_at: OptionalType[datetime] = Field(
        default=None, description="Creation timestamp"
    )
    updated_at: OptionalType[datetime] = Field(
        default=None, description="Last update timestamp"
    )
    deleted_at: OptionalType[datetime] = Field(
        default=None, description="Deletion timestamp (soft delete)"
    )

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
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat(), PydanticObjectId: str},
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


class ForesightRecordProjection(BaseModel):
    """
    Simplified foresight model (without vector)

    Used in most scenarios where vector data is not needed, reducing data transfer and memory usage.
    Note: This model is used for data retrieval only (not stored directly).
    """

    # ID field and audit fields
    id: OptionalType[PydanticObjectId] = Field(
        default=None, description="Document ID"
    )
    created_at: OptionalType[datetime] = Field(
        default=None, description="Creation timestamp"
    )
    updated_at: OptionalType[datetime] = Field(
        default=None, description="Last update timestamp"
    )

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
