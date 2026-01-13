"""
MemCell Lite Model - Minimal MongoDB Storage

Lightweight MemCell model containing only indexed and query fields.
Used to minimize MongoDB storage - only stores indexed fields for queries.
Complete MemCell data is stored in KV-Storage (encrypted).
"""

from datetime import datetime
from typing import List, Optional
from enum import Enum

from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId


class DataTypeEnum(str, Enum):
    """Data type enumeration"""

    CONVERSATION = "Conversation"


class MemCellLite(DocumentBase):
    """
    MemCell Lite Model - Minimal storage version

    Contains only indexed and query fields for MongoDB.
    Full MemCell data is stored in KV-Storage as backup.

    Note: Does not inherit from AuditBase - created_at/updated_at are only
    maintained in the full MemCell stored in KV-Storage.
    """

    # Core indexed fields
    user_id: Optional[Indexed(str)] = Field(
        None,
        description="User ID, core query field. None for group memory, user ID for personal memory",
    )
    timestamp: Indexed(datetime) = Field(..., description="Occurrence time, shard key")

    # Additional query fields
    group_id: Optional[Indexed(str)] = Field(
        default=None, description="Group ID, empty means private chat"
    )
    participants: Optional[List[str]] = Field(
        default=None, description="Names of event participants"
    )
    type: Optional[DataTypeEnum] = Field(default=None, description="Scenario type")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords")

    model_config = ConfigDict(
        # Collection name (same as full MemCell)
        collection="memcells",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "memcells"

        # Same indexes as full MemCell
        indexes = [
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_user_timestamp",
            ),
            IndexModel(
                [("group_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_group_timestamp",
            ),
            IndexModel(
                [("participants", ASCENDING)], name="idx_participants", sparse=True
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("type", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_user_type_timestamp",
            ),
            IndexModel(
                [
                    ('group_id', ASCENDING),
                    ("type", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_group_type_timestamp",
            ),
        ]

        validate_on_save = True
        use_state_management = True


__all__ = ["MemCellLite", "DataTypeEnum"]
