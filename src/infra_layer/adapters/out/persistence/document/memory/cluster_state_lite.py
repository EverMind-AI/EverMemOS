"""
ClusterState Lite Model - Minimal MongoDB Storage

Lightweight ClusterState model containing only indexed and query fields.
Used to minimize MongoDB storage - only stores indexed fields for queries.
Complete ClusterState data is stored in KV-Storage.
"""

from datetime import datetime
from typing import Optional
from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.audit_base import AuditBase
from pydantic import Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from beanie import PydanticObjectId


class ClusterStateLite(DocumentBase, AuditBase):
    """
    ClusterState Lite Model - Minimal storage version

    Contains only indexed and query fields for MongoDB.
    Full ClusterState data is stored in KV-Storage as backup.

    Note: Inherits from AuditBase to automatically manage created_at/updated_at timestamps.
    These audit fields are stored in both MongoDB (for queries) and KV-Storage (for full data).
    """

    # Core indexed field
    group_id: str = Field(..., description="Group ID, primary key")

    model_config = ConfigDict(
        # Collection name (same as full ClusterState)
        collection="cluster_states",
        validate_assignment=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
    )

    class Settings:
        """Beanie settings"""

        name = "cluster_states"

        # Indexes for query fields
        indexes = [
            # Group ID index (primary query field)
            IndexModel([("group_id", ASCENDING)], name="idx_group_id", unique=True),
            # Index on created_at (used by devops scripts for data sync)
            IndexModel([("created_at", DESCENDING)], name="idx_created_at"),
        ]

        validate_on_save = True
        use_state_management = True


__all__ = ["ClusterStateLite"]
