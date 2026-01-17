"""
UserProfile Lite Model - Minimal MongoDB Storage

Lightweight UserProfile model containing only indexed and query fields.
Used to minimize MongoDB storage - only stores indexed fields for queries.
Complete UserProfile data is stored in KV-Storage.
"""

from datetime import datetime
from typing import Optional
from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId


class UserProfileLite(DocumentBase, AuditBase):
    """
    UserProfile Lite Model - Minimal storage version

    Contains only indexed and query fields for MongoDB.
    Full UserProfile data is stored in KV-Storage.

    Note: Inherits from AuditBase to automatically manage created_at/updated_at timestamps.
    These audit fields are stored in both MongoDB (for queries) and KV-Storage (for full data).
    """

    # Core indexed fields
    user_id: str = Field(..., description="User ID")
    group_id: str = Field(..., description="Group ID")

    model_config = ConfigDict(
        collection="user_profiles",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
    )

    class Settings:
        """Beanie settings"""

        name = "user_profiles"

        # Indexes for query fields
        indexes = [
            # Composite index (primary query field)
            IndexModel(
                [("user_id", ASCENDING), ("group_id", ASCENDING)],
                name="idx_user_group",
                unique=True,
            ),
            # Index on user_id (used by get_all_by_user queries)
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            # Index on created_at (used by devops scripts for data sync)
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
        ]

        validate_on_save = True
        use_state_management = True


__all__ = ["UserProfileLite"]
