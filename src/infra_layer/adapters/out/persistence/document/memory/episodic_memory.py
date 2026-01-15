from datetime import datetime
from token import OP
from typing import List, Optional, Dict, Any
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase
from beanie import PydanticObjectId


class EpisodicMemory(DocumentBase, AuditBase):
    """
    Episodic memory document model

    Stores user's episodic memories, including event summaries, participants, topics, etc.
    Directly transferred from MemCell summaries.
    """

    user_id: Optional[str] = Field(
        default=None, description="The individual involved, None indicates group memory"
    )
    user_name: Optional[str] = Field(default=None, description="Name of the individual")
    group_id: Optional[str] = Field(default=None, description="Group ID")
    group_name: Optional[str] = Field(default=None, description="Group name")
    timestamp: datetime = Field(..., description="Occurrence time (timestamp)")
    participants: Optional[List[str]] = Field(
        default=None, description="Names of event participants"
    )
    summary: str = Field(..., min_length=1, description="Memory unit")
    subject: Optional[str] = Field(default=None, description="Memory unit subject")
    episode: str = Field(..., min_length=1, description="Episodic memory")
    type: Optional[str] = Field(
        default=None, description="Episode type, such as Conversation"
    )
    keywords: Optional[List[str]] = Field(default=None, description="Keywords")
    linked_entities: Optional[List[str]] = Field(
        default=None, description="Associated entity IDs"
    )

    memcell_event_id_list: Optional[List[str]] = Field(
        default=None, description="Memory unit event ID"
    )

    parent_type: Optional[str] = Field(
        default=None, description="Parent memory type (e.g., memcell)"
    )
    parent_id: Optional[str] = Field(default=None, description="Parent memory ID")

    extend: Optional[Dict[str, Any]] = Field(
        default=None, description="Reserved extension field"
    )

    vector: Optional[List[float]] = Field(default=None, description="Text vector")
    vector_model: Optional[str] = Field(
        default=None, description="Vectorization model used"
    )

    model_config = ConfigDict(
        collection="episodic_memories",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "group_id": "group_work",
                "timestamp": 1701388800,
                "participants": ["Zhang San", "Li Si"],
                "summary": "Discussed project progress and next week's plan",
                "subject": "Project meeting",
                "episode": "Held a project progress discussion in the meeting room, confirmed next week's development task assignments",
                "type": "Conversation",
                "keywords": ["project", "progress", "meeting"],
                "linked_entities": ["proj_001", "task_123"],
                "extend": {"priority": "high", "location": "Meeting Room A"},
            }
        },
        extra="allow",
    )

    @property
    def event_id(self) -> Optional[PydanticObjectId]:
        return self.id

    class Settings:
        """Beanie settings"""

        name = "episodic_memories"
        indexes = [
            # Single field indexes
            IndexModel([("user_id", ASCENDING)], name="idx_user_id"),
            # Composite index on user ID and timestamp
            IndexModel([("parent_id", ASCENDING)], name="idx_parent_id"),
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_user_timestamp",
            ),
            IndexModel(
                [("group_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_group_timestamp",
                sparse=True,
            ),
            # Composite index on group ID, user ID and timestamp
            # Note: This also covers (group_id, user_id) queries by left-prefix rule
            IndexModel(
                [
                    ("group_id", ASCENDING),
                    ("user_id", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_group_user_timestamp",
                sparse=True,
            ),
            # Index on keywords
            IndexModel([("keywords", ASCENDING)], name="idx_keywords", sparse=True),
            # Index on linked entities
            IndexModel(
                [("linked_entities", ASCENDING)],
                name="idx_linked_entities",
                sparse=True,
            ),
            # Index on audit fields
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
            IndexModel([("updated_at", DESCENDING)], name="idx_updated_at"),
        ]
        validate_on_save = True
        use_state_management = True
