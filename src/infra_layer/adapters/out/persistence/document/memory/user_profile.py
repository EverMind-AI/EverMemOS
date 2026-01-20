from datetime import datetime
from typing import List, Optional, Dict, Any
from typing import Optional as OptionalType
from pydantic import BaseModel, Field, ConfigDict
from beanie import PydanticObjectId


class UserProfile(BaseModel):
    """
    User profile document model

    Stores user profile information automatically extracted from clustering conversations.
    Note: This model is stored in KV-Storage only (not MongoDB).
    """

    # ID field (managed by MongoDB through UserProfileLite)
    id: OptionalType[PydanticObjectId] = Field(
        default=None, description="Document ID (set after MongoDB insert)"
    )

    # Audit fields (managed by MongoDB through UserProfileLite)
    created_at: OptionalType[datetime] = Field(
        default=None, description="Creation timestamp"
    )
    updated_at: OptionalType[datetime] = Field(
        default=None, description="Last update timestamp"
    )

    # Composite primary key
    user_id: str = Field(..., description="User ID")
    group_id: str = Field(..., description="Group ID")

    # Profile content (stored in JSON format)
    profile_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="User profile data (including role, skills, preferences, personality, etc.)",
    )

    # Metadata
    scenario: str = Field(
        default="group_chat", description="Scenario type: group_chat or assistant"
    )
    confidence: float = Field(default=0.0, description="Profile confidence score (0-1)")
    version: int = Field(default=1, description="Profile version number")

    # Clustering association
    cluster_ids: List[str] = Field(
        default_factory=list, description="List of associated cluster IDs"
    )
    memcell_count: int = Field(
        default=0, description="Number of MemCells involved in extraction"
    )

    # History
    last_updated_cluster: Optional[str] = Field(
        default=None, description="Cluster ID used in the last update"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat(), PydanticObjectId: str},
        extra="allow",
    )
