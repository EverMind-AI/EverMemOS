"""
ForesightRecord Lite Model - Minimal MongoDB Storage

Lightweight ForesightRecord model containing only indexed and query fields.
Used to minimize MongoDB storage - only stores indexed fields for queries.
Complete ForesightRecord data is stored in KV-Storage.
"""

from datetime import datetime
from typing import Optional
from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId


class ForesightRecordLite(DocumentBase, AuditBase):
    """
    ForesightRecord Lite Model - Minimal storage version

    Contains only indexed and query fields for MongoDB.
    Full ForesightRecord data is stored in KV-Storage as backup.

    Note: Inherits from AuditBase to automatically manage created_at/updated_at timestamps.
    These audit fields are stored in both MongoDB (for queries) and KV-Storage (for full data).
    """

    # Core indexed fields
    user_id: Optional[str] = Field(
        default=None,
        description="User ID, required for personal memory, None for group memory",
    )
    group_id: Optional[str] = Field(default=None, description="Group ID")
    parent_episode_id: str = Field(..., description="Parent episodic memory's event_id")

    model_config = ConfigDict(
        # Collection name (same as full ForesightRecord)
        collection="foresight_records",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "foresight_records"

        # Indexes for query fields
        indexes = [
            # User ID index
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            # Parent episodic memory index
            IndexModel([("parent_episode_id", ASCENDING)], name="idx_parent_episode"),
            # Composite index of user ID and parent episodic memory
            IndexModel(
                [("user_id", ASCENDING), ("parent_episode_id", ASCENDING)],
                name="idx_user_parent",
            ),
            # Group ID index
            IndexModel([("group_id", ASCENDING)], name="idx_group_id", sparse=True),
            # Index on created_at (used by devops scripts for data sync)
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
        ]

        validate_on_save = True
        use_state_management = True


__all__ = ["ForesightRecordLite"]
