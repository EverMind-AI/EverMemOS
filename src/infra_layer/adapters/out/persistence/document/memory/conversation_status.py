from datetime import datetime
from typing import Optional
from typing import Optional as OptionalType
from pydantic import BaseModel, Field, ConfigDict
from beanie import PydanticObjectId


class ConversationStatus(BaseModel):
    """
    Conversation status document model

    Stores conversation status information, including group ID, message read time, etc.

    Note: This model is stored in KV-Storage only (not MongoDB).
    """

    # ID field (managed by MongoDB through ConversationStatusLite)
    id: OptionalType[PydanticObjectId] = Field(
        default=None, description="Document ID (set after MongoDB insert)"
    )

    # Audit fields (managed by MongoDB through ConversationStatusLite)
    created_at: OptionalType[datetime] = Field(
        default=None, description="Creation timestamp"
    )
    updated_at: OptionalType[datetime] = Field(
        default=None, description="Last update timestamp"
    )

    # Basic information
    group_id: str = Field(..., description="Group ID, empty means private chat")
    old_msg_start_time: Optional[datetime] = Field(
        default=None, description="Conversation window read start time"
    )
    new_msg_start_time: Optional[datetime] = Field(
        default=None, description="Accumulated new conversation read start time"
    )
    last_memcell_time: Optional[datetime] = Field(
        default=None, description="Accumulated memCell read start time"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat(), PydanticObjectId: str},
        extra="allow",
    )

    @property
    def conversation_id(self) -> Optional[PydanticObjectId]:
        return self.id
