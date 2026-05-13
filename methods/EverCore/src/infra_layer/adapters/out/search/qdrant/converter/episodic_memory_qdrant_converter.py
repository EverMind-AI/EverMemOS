"""
Episodic Memory Qdrant Converter.

Converts MongoDB ``v1_episodic_memories`` documents to Qdrant ``PointStruct``
instances for upsert into ``v1_episodic_memory``. Only search-essential
fields are mapped — full payload is fetched from MongoDB via ``parent_id``
back-reference.

Vector is taken from ``source_doc.vector`` (caller must have populated the
embedding before calling the converter).
"""

import json
from typing import List

from qdrant_client.http import models as qmodels

from core.observation.logger import get_logger
from core.oxm.qdrant.base_converter import BaseQdrantConverter
from core.oxm.qdrant.base_repository import mongo_id_to_qdrant_id
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory as MongoEpisodicMemory,
)
from infra_layer.adapters.out.search.qdrant.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)

logger = get_logger(__name__)


class EpisodicMemoryQdrantConverter(BaseQdrantConverter[EpisodicMemoryCollection]):
    """
    Converts MongoDB ``v1_episodic_memories`` documents to Qdrant point payloads.

    Output shape: ``qdrant_client.http.models.PointStruct`` with the document
    id as point id, the pre-computed embedding as the vector, and all
    search-relevant scalar fields plus the MongoDB back-reference in the
    payload.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEpisodicMemory) -> qmodels.PointStruct:
        """
        Build a ``PointStruct`` from a MongoDB episodic-memory document.

        Args:
            source_doc: MongoDB ``v1_episodic_memories`` document instance.

        Returns:
            ``PointStruct`` ready for ``client.upsert([point])``.

        Raises:
            ValueError: when ``source_doc`` is ``None``.
            Exception: on any conversion failure (logged + re-raised).
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be None")
        if source_doc.id is None:
            raise ValueError("EpisodicMemory.id must not be None")

        try:
            # Timestamp -> epoch milliseconds (integer, parity with Milvus).
            timestamp_ms = (
                int(source_doc.timestamp.timestamp() * 1000)
                if source_doc.timestamp
                else 0
            )

            search_content = cls._build_search_content(source_doc)

            payload = {
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "session_id": source_doc.session_id or "",
                "participants": source_doc.participants or [],
                "sender_ids": getattr(source_doc, "sender_ids", []) or [],
                "type": getattr(source_doc, "type", None) or "",
                "timestamp": timestamp_ms,
                "episode": source_doc.episode or "",
                "search_content": search_content,
                "parent_type": source_doc.parent_type or "",
                "parent_id": (
                    str(source_doc.parent_id) if source_doc.parent_id else ""
                ),
                # Mongo back-reference: Qdrant ids are derived via uuid5, so
                # we keep the raw Mongo id in the payload for round-trip
                # lookup, idempotent re-embed, and debugging.
                "mongo_id": str(source_doc.id),
            }

            vector = (
                source_doc.vector
                if hasattr(source_doc, "vector") and source_doc.vector
                else None
            )
            if not vector:
                raise ValueError(
                    f"Vector is required for EpisodicMemory {source_doc.id} "
                    "but was not populated"
                )

            return qmodels.PointStruct(
                id=mongo_id_to_qdrant_id(source_doc.id),
                vector=vector,
                payload=payload,
            )

        except Exception as e:
            logger.exception(
                "Failed to convert MongoDB document to Qdrant point: %s", e
            )
            raise

    @staticmethod
    def _build_search_content(source_doc: MongoEpisodicMemory) -> str:
        """
        Build search content string from the document's text fields.

        Returns a JSON-stringified list (parity with Milvus converter; the
        search pipeline can deserialize it back to a list when needed).
        """
        text_content: List[str] = []

        if hasattr(source_doc, "subject") and source_doc.subject:
            text_content.append(source_doc.subject)

        if hasattr(source_doc, "summary") and source_doc.summary:
            text_content.append(source_doc.summary)

        if hasattr(source_doc, "episode") and source_doc.episode:
            text_content.append(source_doc.episode)

        return json.dumps(text_content, ensure_ascii=False)
