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
from core.oxm.mongo.document_base_with_soft_delete import DocumentBaseWithSoftDelete
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId


class DataTypeEnum(str, Enum):
    """Data type enumeration"""

    CONVERSATION = "Conversation"


class MemCellLite(DocumentBaseWithSoftDelete, AuditBase):
    """
    MemCell Lite Model - Minimal storage version

    Contains only indexed and query fields for MongoDB.
    Full MemCell data is stored in KV-Storage as backup.

    Supports soft delete functionality:
    - Use delete() method for soft deletion
    - Use find_one(), find_many() to automatically filter out deleted records
    - Use hard_find_one(), hard_find_many() to query including deleted records
    - Use hard_delete() for physical deletion

    Note: Inherits from AuditBase to automatically manage created_at/updated_at timestamps.
    These audit fields are stored in both MongoDB (for queries) and KV-Storage (for full data).
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

        # Index definitions (matching main branch MemCell)
        # MemCellLite is stored in MongoDB with all the same indexes as main branch
        indexes = [
            # 1. Soft delete support - soft delete status index
            IndexModel(
                [("deleted_at", ASCENDING)],
                name="idx_deleted_at",
                sparse=True,  # Only index documents that are deleted
            ),
            # 2. Composite index for user queries - core query pattern
            # Includes deleted_at to optimize soft delete filtering
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_user_deleted_timestamp",
            ),
            # 3. Composite index for group queries - optimized for group chat scenarios
            # Includes deleted_at to optimize soft delete filtering
            IndexModel(
                [
                    ("group_id", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_group_deleted_timestamp",
            ),
            # 4. Index for time range queries (shard key, automatically created by MongoDB)
            # Note: Shard key index is automatically created, no need to define manually
            # IndexModel([("timestamp", ASCENDING)], name="idx_timestamp"),
            # 5. Index for participant queries - indexing multi-value field
            IndexModel(
                [("participants", ASCENDING)], name="idx_participants", sparse=True
            ),
            # 6. Composite index for user-type queries - optimized for user data type filtering
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("type", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_user_type_deleted_timestamp",
            ),
            # 7. Composite index for group-type queries - optimized for group data type filtering
            IndexModel(
                [
                    ('group_id', ASCENDING),
                    ("type", ASCENDING),
                    ("deleted_at", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_group_type_deleted_timestamp",
            ),
            # Creation time index
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            # Update time index
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]

        validate_on_save = True
        use_state_management = True


__all__ = ["MemCellLite", "DataTypeEnum"]
