"""
AgentCase Qdrant Converter.

Converts MongoDB ``AgentCaseRecord`` documents into Qdrant ``PointStruct``
instances for upsert into ``v1_agent_case``. Vector is the embedding of
``task_intent`` (caller must populate ``source_doc.vector`` first).
"""

from qdrant_client.http import models as qmodels

from core.observation.logger import get_logger
from core.oxm.qdrant.base_converter import BaseQdrantConverter
from core.oxm.qdrant.base_repository import mongo_id_to_qdrant_id
from infra_layer.adapters.out.persistence.document.memory.agent_case import (
    AgentCaseRecord,
)
from infra_layer.adapters.out.search.qdrant.memory.agent_case_collection import (
    AgentCaseCollection,
)

logger = get_logger(__name__)


class AgentCaseQdrantConverter(BaseQdrantConverter[AgentCaseCollection]):
    """Converts MongoDB ``AgentCaseRecord`` documents into Qdrant point payloads."""

    @classmethod
    def from_mongo(cls, source_doc: AgentCaseRecord) -> qmodels.PointStruct:
        """
        Build a ``PointStruct`` from a MongoDB ``AgentCaseRecord``.

        Raises:
            ValueError: when ``source_doc`` is ``None``.
            Exception: on any conversion failure (logged + re-raised).
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be None")
        if source_doc.id is None:
            raise ValueError("AgentCaseRecord.id must not be None")

        try:
            vector = source_doc.vector if source_doc.vector else None
            if not vector:
                raise ValueError(
                    f"Vector is required for AgentCaseRecord {source_doc.id} "
                    "but was not populated"
                )

            task_intent = source_doc.task_intent or ""
            # Parity with Milvus converter: epoch seconds (not ms) for this collection.
            timestamp_s = (
                int(source_doc.timestamp.timestamp()) if source_doc.timestamp else 0
            )

            payload = {
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "session_id": source_doc.session_id or "",
                "timestamp": timestamp_s,
                "task_intent": task_intent[:5000],
                "parent_type": source_doc.parent_type or "",
                "parent_id": source_doc.parent_id or "",
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
                "Failed to convert AgentCaseRecord to Qdrant point: %s", e
            )
            raise
