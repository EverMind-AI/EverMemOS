from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field
from core.oxm.mongo.audit_base import AuditBase


class ClusterState(DocumentBase, AuditBase):
    """
    Cluster state document model

    Each group maintains a cluster state document, storing complete clustering information for that group
    """

    # Primary key
    group_id: Indexed(str) = Field(..., description="Group ID, primary key")

    # Basic clustering information
    event_ids: List[str] = Field(
        default_factory=list, description="List of all event_ids"
    )
    timestamps: List[float] = Field(
        default_factory=list, description="List of timestamps"
    )
    cluster_ids: List[str] = Field(
        default_factory=list, description="List of cluster IDs"
    )

    # event_id -> cluster_id mapping
    eventid_to_cluster: Dict[str, str] = Field(
        default_factory=dict, description="Mapping from event to cluster"
    )

    # Clustering metadata
    next_cluster_idx: int = Field(default=0, description="Next cluster index")

    # Cluster centroid information (vectors stored as lists)
    cluster_centroids: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Cluster centroid vectors {cluster_id: vector}",
    )
    cluster_counts: Dict[str, int] = Field(
        default_factory=dict, description="Cluster member count {cluster_id: count}"
    )
    cluster_last_ts: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Cluster last timestamp {cluster_id: timestamp}",
    )

    class Settings:
        name = "cluster_states"

        # Dual Storage architecture:
        # - MongoDB stores ClusterStateLite (indexed fields only)
        # - KV-Storage stores complete ClusterState (full data)
