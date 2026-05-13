"""
Atomic Fact Qdrant Converter.

Converts MongoDB ``v1_atomic_fact_records`` documents to Qdrant
``PointStruct`` instances for upsert into ``v1_atomic_fact_record``.
"""

import json
from typing import List

from qdrant_client.http import models as qmodels

from api_specs.memory_types import RawDataType
from core.observation.logger import get_logger
from core.oxm.qdrant.base_converter import BaseQdrantConverter
from core.oxm.qdrant.base_repository import mongo_id_to_qdrant_id
from infra_layer.adapters.out.persistence.document.memory.atomic_fact_record import (
    AtomicFactRecord as MongoAtomicFactRecord,
)
from infra_layer.adapters.out.search.qdrant.memory.atomic_fact_collection import (
    AtomicFactCollection,
)

logger = get_logger(__name__)


class AtomicFactQdrantConverter(BaseQdrantConverter[AtomicFactCollection]):
    """Converts MongoDB ``v1_atomic_fact_records`` to Qdrant point payloads."""

    @classmethod
    def from_mongo(cls, source_doc: MongoAtomicFactRecord) -> qmodels.PointStruct:
        """
        Build a ``PointStruct`` from a MongoDB atomic-fact document.

        Raises:
            ValueError: when ``source_doc`` is ``None``.
            Exception: on any conversion failure (logged + re-raised).
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be None")
        if source_doc.id is None:
            raise ValueError("AtomicFactRecord.id must not be None")

        try:
            timestamp_ms = (
                int(source_doc.timestamp.timestamp() * 1000)
                if source_doc.timestamp
                else 0
            )

            # ``getattr(... , None)`` then explicit ``is None`` check so a
            # legitimately falsy value (e.g. empty string from a future
            # type enum entry) is preserved.
            raw_type = getattr(source_doc, "type", None)
            event_type = (
                raw_type if raw_type is not None else RawDataType.CONVERSATION.value
            )

            payload = {
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "session_id": source_doc.session_id or "",
                "participants": source_doc.participants or [],
                "sender_ids": getattr(source_doc, "sender_ids", []) or [],
                "type": event_type,
                "timestamp": timestamp_ms,
                "parent_type": source_doc.parent_type or "",
                "parent_id": (
                    str(source_doc.parent_id) if source_doc.parent_id else ""
                ),
                # Persist the canonical text so search results can return the
                # underlying atomic_fact without a Mongo round-trip.
                "search_content": cls._build_search_content(source_doc),
                # Mongo back-reference (see episodic_memory converter).
                "mongo_id": str(source_doc.id),
            }

            vector = getattr(source_doc, "vector", None)
            if not vector:
                raise ValueError(
                    f"Vector is required for AtomicFactRecord {source_doc.id} "
                    "but was not populated"
                )

            return qmodels.PointStruct(
                id=mongo_id_to_qdrant_id(source_doc.id),
                vector=vector,
                payload=payload,
            )

        except Exception as e:
            logger.exception(
                "Failed to convert MongoDB AtomicFact to Qdrant point: %s", e
            )
            raise

    @staticmethod
    def _build_search_content(source_doc: MongoAtomicFactRecord) -> str:
        """Build search content JSON-string from the atomic_fact text field."""
        text_content: List[str] = []
        if source_doc.atomic_fact:
            text_content.append(source_doc.atomic_fact)
        return json.dumps(text_content, ensure_ascii=False)
