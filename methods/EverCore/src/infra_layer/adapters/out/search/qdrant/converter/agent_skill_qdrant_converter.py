"""
AgentSkill Qdrant Converter.

Converts MongoDB ``AgentSkillRecord`` documents into Qdrant ``PointStruct``
instances for upsert into ``v1_agent_skill``. Vector is the embedding of
name + description (caller-provided via ``source_doc.vector``).
"""

from qdrant_client.http import models as qmodels

from core.observation.logger import get_logger
from core.oxm.qdrant.base_converter import BaseQdrantConverter
from core.oxm.qdrant.base_repository import mongo_id_to_qdrant_id
from infra_layer.adapters.out.persistence.document.memory.agent_skill import (
    AgentSkillRecord,
)
from infra_layer.adapters.out.search.qdrant.memory.agent_skill_collection import (
    AgentSkillCollection,
)

logger = get_logger(__name__)


class AgentSkillQdrantConverter(BaseQdrantConverter[AgentSkillCollection]):
    """Converts MongoDB ``AgentSkillRecord`` documents into Qdrant point payloads."""

    @classmethod
    def from_mongo(cls, source_doc: AgentSkillRecord) -> qmodels.PointStruct:
        """
        Build a ``PointStruct`` from a MongoDB ``AgentSkillRecord``.

        Raises:
            ValueError: when ``source_doc`` is ``None``.
            Exception: on any conversion failure (logged + re-raised).
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be None")
        if source_doc.id is None:
            raise ValueError("AgentSkillRecord.id must not be None")

        try:
            vector = source_doc.vector if source_doc.vector else None
            if not vector:
                raise ValueError(
                    f"Vector is required for AgentSkillRecord {source_doc.id} "
                    "but was not populated"
                )

            name = source_doc.name or ""
            description = source_doc.description or ""

            # Primary text field: name + newline + description (Milvus parity).
            content_field = "\n".join(s for s in [name, description] if s)

            payload = {
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "cluster_id": source_doc.cluster_id or "",
                "content": content_field[:5000],
                # Coerce optional scores to 0.0 — Qdrant range-filters silently
                # exclude ``null``-valued payloads, which would hide scored points
                # from threshold queries. Treat "absent" as "lowest score".
                "maturity_score": (
                    source_doc.maturity_score
                    if source_doc.maturity_score is not None
                    else 0.0
                ),
                "confidence": (
                    source_doc.confidence
                    if source_doc.confidence is not None
                    else 0.0
                ),
                # Mongo back-reference (see episodic_memory converter).
                "mongo_id": str(source_doc.id),
            }

            return qmodels.PointStruct(
                id=mongo_id_to_qdrant_id(source_doc.id),
                vector=vector,
                payload=payload,
            )

        except Exception as e:
            logger.exception(
                "Failed to convert AgentSkillRecord to Qdrant point: %s", e
            )
            raise
