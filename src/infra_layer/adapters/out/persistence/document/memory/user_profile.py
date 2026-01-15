from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field
from core.oxm.mongo.audit_base import AuditBase
from pymongo import IndexModel, ASCENDING, DESCENDING


class UserProfile(DocumentBase, AuditBase):
    """
    User profile document model

    Stores user profile information automatically extracted from clustering conversations
    """

    # Composite primary key
    user_id: Indexed(str) = Field(..., description="User ID")
    group_id: Indexed(str) = Field(..., description="Group ID")

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

    class Settings:
        """Beanie settings"""

        name = "user_profiles"
        indexes = [
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            IndexModel([("group_id", ASCENDING)], name="idx_group_id"),
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
