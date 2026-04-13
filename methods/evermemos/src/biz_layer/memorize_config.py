"""
Memory retrieval process configuration

Centralized management of all trigger conditions and thresholds for easy adjustment and maintenance.
"""

import os
from dataclasses import dataclass

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
    # Profile extraction interval: extract once every N memcells (1 = every time)
    profile_extraction_interval: int = 1
    # Minimum confidence required for Profile extraction
    profile_min_confidence: float = 0.6
    # Whether to enable version control
    profile_enable_versioning: bool = True
    # Profile maximum items
    profile_max_items: int = 25

    # ===== Parent type configuration =====
    # Default parent type for Episode (memcell or episode)
    default_episode_parent_type: str = ParentType.MEMCELL.value
    # Default parent type for Foresight (memcell or episode)
    default_foresight_parent_type: str = ParentType.MEMCELL.value
    # Default parent type for AtomicFact (memcell or episode)
    default_atomic_fact_parent_type: str = ParentType.MEMCELL.value

    # ===== Agent Skill extraction configuration =====
    # Minimum quality score (0.0-1.0) of the AgentCase required to trigger
    # skill extraction. Cases below this threshold are considered too low
    # quality to contribute to skill formation.
    skill_min_quality_score: float = 0.2
    # Minimum maturity score (0.0-1.0) for a skill to be retrievable
    skill_maturity_threshold: float = 0.6
    # Minimum confidence (0.0-1.0) for a skill to remain active.
    # Skills whose confidence drops below this threshold are kept in MongoDB
    # (data preserved) but removed from search engines and excluded from
    # future extraction context.
    skill_retire_confidence: float = 0.1
    # Skip LLM-based maturity scoring for skills. When True, all skills
    # are assigned maturity_score=1.0 directly, saving one LLM call per
    # add/update operation.
    skip_skill_maturity_scoring: bool = False
    # Skip foresight and atomic_fact extraction for agent conversations.
    # When True, only episodes and agent_case are extracted, saving LLM
    # calls that are not needed for the skill extraction pipeline.
    skip_foresight_and_eventlog: bool = False

    # ===== Extraction toggles (for fast evaluation) =====
    # When True, skip agent skill extraction entirely.
    skip_skill_extraction: bool = False
    # When True, skip profile extraction entirely.
    skip_profile_extraction: bool = False

    # ===== Skill retrieval configuration =====
    # When True, apply LLM-based relevance verification after vector search
    # for agent skills, filtering out irrelevant results.
    enable_skill_llm_verify: bool = False

    # ===== LLM request configuration =====
    # When True, disable reasoning/thinking for episode and agent case
    # extraction by injecting {"chat_template_kwargs": {"enable_thinking": false}}
    # into every LLM request body. Useful for reasoning models (e.g. Qwen3.5)
    # deployed via vLLM or SGLang.
    skip_episode_case_reasoning: bool = False
    # When True, disable reasoning/thinking for LLM-based clustering.
    skip_clustering_reasoning: bool = False

    # ===== Batch clustering configuration =====
    # Accumulate N memcells before running clustering. 1 = cluster immediately
    # (current behavior). Values > 1 reduce lock contention and enable batched
    # embedding calls. Use the flush-clustering API to drain pending items on demand.
    cluster_batch_size: int = 1



# Global default configuration (can be overridden via from_env())
# TODO Move nescessary configurations to ENV. Use default values for now.
DEFAULT_MEMORIZE_CONFIG = MemorizeConfig()

_agent_cluster_similarity_threshold = float(os.getenv("AGENT_CLUSTER_SIMILARITY_THRESHOLD", "0.5"))

FAST_SKILL_MEMORIZE_CONFIG = MemorizeConfig(
    cluster_similarity_threshold=_agent_cluster_similarity_threshold,
    cluster_batch_size=int(os.getenv("AGENT_CLUSTER_BATCH_SIZE", "20")),
    skip_skill_maturity_scoring=True,
    skip_foresight_and_eventlog=True,
    skip_profile_extraction=True,
    enable_skill_llm_verify=True,
    skip_episode_case_reasoning=True,
)

ONLINE_AGENT_MEMORIZE_CONFIG = MemorizeConfig(
    cluster_similarity_threshold=_agent_cluster_similarity_threshold,
)

_agent_mode = os.getenv("AGENT_MEMORIZE_MODE", "online").lower()
if _agent_mode == "fast_skill":
    AGENT_DEFAULT_MEMORIZE_CONFIG = FAST_SKILL_MEMORIZE_CONFIG
else:
    AGENT_DEFAULT_MEMORIZE_CONFIG = ONLINE_AGENT_MEMORIZE_CONFIG
