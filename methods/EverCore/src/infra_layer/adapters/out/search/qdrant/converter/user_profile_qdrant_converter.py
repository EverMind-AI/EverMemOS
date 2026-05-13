"""
User Profile Qdrant Converter.

Converts a single MongoDB ``v1_user_profiles`` document into a **list** of
profile items — one per ``explicit_info`` entry and one per
``implicit_trait``. Each item carries an ``embed_text`` field used by the
ProfileIndexer to generate the actual embedding; the indexer then wraps
each item into a ``PointStruct`` and upserts it.

Return type intentionally diverges from the other Qdrant converters
(``PointStruct``) — it returns ``List[Dict[str, Any]]`` for parity with the
Milvus counterpart, because the indexer flow expects per-item dicts
(vector is **not yet** set at converter time; that happens downstream).

The ``from_mongo`` override carries ``# type: ignore[override]`` because of
this intentional contract divergence from the generic ``BaseQdrantConverter``
signature. The ProfileIndexer downstream is the only known consumer.
"""

from typing import Any, Dict, List

from api_specs.memory_types import ScenarioType
from core.observation.logger import get_logger
from core.oxm.mongo.mongo_utils import generate_object_id_str
from core.oxm.qdrant.base_converter import BaseQdrantConverter
from infra_layer.adapters.out.persistence.document.memory.user_profile import (
    UserProfile as MongoUserProfile,
)
from infra_layer.adapters.out.search.qdrant.memory.user_profile_collection import (
    UserProfileCollection,
)

logger = get_logger(__name__)


# Profile-data shape (matches Milvus converter):
# explicit fields contain [{value, level?}] items (skills / responsibilities / interests).
_EXPLICIT_FIELDS = [
    ("hard_skills", "Hard Skill"),
    ("soft_skills", "Soft Skill"),
    ("work_responsibility", "Work Responsibility"),
    ("interests", "Interest"),
]

# Implicit fields contain [{value}] items (personality / tendencies / values).
_IMPLICIT_FIELDS = [
    ("personality", "Personality"),
    ("tendency", "Tendency"),
    ("way_of_decision_making", "Decision Making"),
    ("motivation_system", "Motivation"),
    ("fear_system", "Fear"),
    ("value_system", "Value"),
]


class UserProfileQdrantConverter(BaseQdrantConverter[UserProfileCollection]):
    """Splits a MongoDB ``UserProfile`` doc into per-item Qdrant payload dicts."""

    @classmethod
    def from_mongo(  # type: ignore[override]
        cls, source_doc: MongoUserProfile
    ) -> List[Dict[str, Any]]:
        """
        Convert a single ``UserProfile`` doc into a list of per-item payloads.

        The returned list contains one dict per ``explicit_info`` /
        ``implicit_trait`` entry. Each dict has:
        - ``id``: a freshly generated ObjectId string (the Mongo doc ``id``
          would collide across items because we emit many points from one
          source doc).
        - All filterable payload fields (user_id, group_id, scenario,
          memcell_count, item_type).
        - ``embed_text``: the text used by the ProfileIndexer to generate
          the embedding vector. The vector is **not** included — the
          indexer wraps the dict into ``PointStruct`` after embedding.

        Raises:
            ValueError: when ``source_doc`` is ``None``.
            Exception: on any conversion failure (logged + re-raised).
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            profile_data: Dict[str, Any] = source_doc.profile_data or {}
            user_id = source_doc.user_id or ""
            group_id = source_doc.group_id or ""
            scenario = source_doc.scenario or ScenarioType.SOLO.value
            memcell_count = source_doc.memcell_count or 0

            items: List[Dict[str, Any]] = []

            def _make_item(embed_text: str, item_type: str) -> Dict[str, Any]:
                return {
                    "id": generate_object_id_str(),
                    "user_id": user_id,
                    "group_id": group_id,
                    "scenario": scenario,
                    "memcell_count": memcell_count,
                    "item_type": item_type,
                    "embed_text": embed_text,
                }

            # ProfileMemory format: per-field lists of {value, level?, ...}.
            for field_name, label in _EXPLICIT_FIELDS:
                for entry in profile_data.get(field_name, []) or []:
                    value = (
                        entry.get("value", "")
                        if isinstance(entry, dict)
                        else str(entry)
                    )
                    if not value:
                        continue
                    level = entry.get("level", "") if isinstance(entry, dict) else ""
                    embed_text = (
                        f"{label}: {value}" + (f" ({level})" if level else "")
                    )
                    items.append(_make_item(embed_text, "explicit_info"))

            for field_name, label in _IMPLICIT_FIELDS:
                for entry in profile_data.get(field_name, []) or []:
                    value = (
                        entry.get("value", "")
                        if isinstance(entry, dict)
                        else str(entry)
                    )
                    if not value:
                        continue
                    items.append(_make_item(f"{label}: {value}", "implicit_trait"))

            # Legacy format: flat explicit_info[] / implicit_traits[] arrays
            # with {category, description} / {trait, description, basis} shape.
            for entry in profile_data.get("explicit_info", []) or []:
                if not isinstance(entry, dict):
                    continue
                desc = entry.get("description", "")
                if not desc:
                    continue
                category = entry.get("category", "")
                embed_text = f"{category}: {desc}" if category else desc
                items.append(_make_item(embed_text, "explicit_info"))

            for entry in profile_data.get("implicit_traits", []) or []:
                if not isinstance(entry, dict):
                    continue
                desc = entry.get("description", "")
                if not desc:
                    continue
                trait_name = entry.get("trait") or entry.get("trait_name", "")
                embed_text = f"{trait_name}: {desc}" if trait_name else desc
                if entry.get("basis"):
                    embed_text += f". {entry['basis']}"
                items.append(_make_item(embed_text, "implicit_trait"))

            # Single user-goal string.
            user_goal = profile_data.get("user_goal")
            if user_goal and isinstance(user_goal, str) and user_goal.strip():
                items.append(
                    _make_item(f"Goal: {user_goal.strip()}", "explicit_info")
                )

            return items

        except Exception as e:
            logger.exception(
                "Failed to convert MongoDB UserProfile to Qdrant items: %s", e
            )
            raise
