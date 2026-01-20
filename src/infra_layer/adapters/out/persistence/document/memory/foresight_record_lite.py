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
    parent_id: str = Field(..., description="Parent memory ID")
    parent_type: Optional[str] = Field(
        default=None, description="Parent memory type (memcell/episode)"
    )

    # Time range fields (for time overlap queries)
    start_time: Optional[str] = Field(
        default=None, description="Foresight start time (date string, e.g., 2024-01-01)"
    )
    end_time: Optional[str] = Field(
        default=None, description="Foresight end time (date string, e.g., 2024-12-31)"
    )

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

        # Indexes for query fields (matching main branch)
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
            # Indexes on audit fields (for pagination and time-based queries)
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]

        validate_on_save = True
        use_state_management = True


__all__ = ["ForesightRecordLite"]
