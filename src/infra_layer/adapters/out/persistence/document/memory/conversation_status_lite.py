"""
ConversationStatus Lite Model - Minimal MongoDB Storage

Lightweight ConversationStatus model containing only indexed and query fields.
Used to minimize MongoDB storage - only stores indexed fields for queries.
Complete ConversationStatus data is stored in KV-Storage.
"""

from datetime import datetime
from typing import Optional
from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId


class ConversationStatusLite(DocumentBase, AuditBase):
    """
    ConversationStatus Lite Model - Minimal storage version

    Contains only indexed and query fields for MongoDB.
    Full ConversationStatus data is stored in KV-Storage.

    Note: Inherits from AuditBase to automatically manage created_at/updated_at timestamps.
    These audit fields are stored in both MongoDB (for queries) and KV-Storage (for full data).
    """

    # Core indexed field
    group_id: str = Field(..., description="Group ID, empty means private chat")

    model_config = ConfigDict(
        collection="conversation_status",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
    )

    @property
    def conversation_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "conversation_status"
        indexes = [
            # Note: conversation_id maps to the _id field, MongoDB automatically creates a primary key index on _id
            IndexModel(
                [("group_id", ASCENDING)], name="idx_group_id", unique=True
            ),  # group_id must be unique
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True


__all__ = ["ConversationStatusLite"]
