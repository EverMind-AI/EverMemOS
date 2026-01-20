"""
CoreMemory Lite Model - Minimal MongoDB Storage

Lightweight CoreMemory model containing only indexed and query fields.
Used to minimize MongoDB storage - only stores indexed fields for queries.
Complete CoreMemory data is stored in KV-Storage.
"""

from datetime import datetime
from typing import Optional
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING


class CoreMemoryLite(DocumentBase, AuditBase):
    """
    CoreMemory Lite Model - Minimal storage version

    Contains only indexed and query fields for MongoDB.
    Full CoreMemory data is stored in KV-Storage as backup.

    Note: Inherits from AuditBase to automatically manage created_at/updated_at timestamps.
    These audit fields are stored in both MongoDB (for queries) and KV-Storage (for full data).
    """

    # Core indexed fields
    user_id: Indexed(str) = Field(..., description="User ID")

    # Version control fields
    version: Optional[str] = Field(
        default=None, description="Version number, used for version management"
    )
    is_latest: Optional[bool] = Field(
        default=True, description="Whether it is the latest version, default is True"
    )

    model_config = ConfigDict(
        # Collection name (same as full CoreMemory)
        collection="core_memories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
    )

    class Settings:
        """Beanie settings"""

        name = "core_memories"

        # Indexes for query fields (audit field indexes included for time-based queries)
        indexes = [
            # Unique compound index on user_id and version
            IndexModel(
                [("user_id", ASCENDING), ("version", ASCENDING)],
                unique=True,
                name="idx_user_id_version_unique",
            ),
            # Index on is_latest field (for fast querying of latest version)
            IndexModel(
                [("user_id", ASCENDING), ("is_latest", ASCENDING)],
                name="idx_user_id_is_latest",
            ),
            # Indexes on audit fields (for pagination and time-based queries)
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]

        validate_on_save = True
        use_state_management = True


__all__ = ["CoreMemoryLite"]
