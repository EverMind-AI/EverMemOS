"""
Memory retrieval process configuration

Centralized management of all trigger conditions and thresholds for easy adjustment and maintenance.
"""

from dataclasses import dataclass
import os

from api_specs.memory_types import ParentType


@dataclass
class MemorizeConfig:
    """Memory retrieval process configuration"""

    # ===== Clustering configuration =====
    # Semantic similarity threshold; memcells exceeding this value will be clustered into the same cluster
    cluster_similarity_threshold: float = 0.3
    # Maximum time gap (days); memcells exceeding this gap will not be clustered together
    cluster_max_time_gap_days: int = 7

    # ===== Profile extraction configuration =====
    # Minimum number of memcells required to trigger Profile extraction
    profile_min_memcells: int = 1
    # Minimum confidence required for Profile extraction
    profile_min_confidence: float = 0.6
    # Whether to enable version control
    profile_enable_versioning: bool = True
    # Life Profile maximum items (ASSISTANT scene only)
    profile_life_max_items: int = 25

    # ===== Foresight/EventLog extraction configuration =====
    # Default parent type for Foresight and EventLog (memcell or episode)
    default_parent_type: str = ParentType.MEMCELL.value

    @classmethod
    def from_env(cls) -> "MemorizeConfig":
        """Load configuration from environment variables, use defaults if not set"""
        return cls(
            cluster_similarity_threshold=float(
                os.getenv("CLUSTER_SIMILARITY_THRESHOLD", "0.3")
            ),
            cluster_max_time_gap_days=int(os.getenv("CLUSTER_MAX_TIME_GAP_DAYS", "7")),
            profile_min_memcells=int(os.getenv("PROFILE_MIN_MEMCELLS", "1")),
            profile_min_confidence=float(os.getenv("PROFILE_MIN_CONFIDENCE", "0.6")),
            profile_enable_versioning=os.getenv(
                "PROFILE_ENABLE_VERSIONING", "true"
            ).lower()
            == "true",
            default_parent_type=os.getenv(
                "DEFAULT_PARENT_TYPE", ParentType.MEMCELL.value
            ),
        )


# Global default configuration (can be overridden via from_env())
# TODO Move nescessary configurations to ENV. Use default values for now.
DEFAULT_MEMORIZE_CONFIG = MemorizeConfig()
