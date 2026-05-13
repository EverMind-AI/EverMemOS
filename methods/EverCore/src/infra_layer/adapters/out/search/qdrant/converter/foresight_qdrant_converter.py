"""
Foresight Qdrant Converter.

Converts MongoDB ``v1_foresight_records`` documents to Qdrant ``PointStruct``
instances for upsert into ``v1_foresight_record``.

Time fields (``start_time``, ``end_time``) accept ``datetime``, ISO-8601
strings, or numeric epoch seconds — all normalized to epoch milliseconds
on output (Milvus parity).
"""

import json
from datetime import datetime
from typing import List, Optional, Union

from qdrant_client.http import models as qmodels

from core.observation.logger import get_logger
from core.oxm.qdrant.base_converter import BaseQdrantConverter
from core.oxm.qdrant.base_repository import mongo_id_to_qdrant_id
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord as MongoForesightRecord,
)
from infra_layer.adapters.out.search.qdrant.memory.foresight_collection import (
    ForesightCollection,
)

logger = get_logger(__name__)


class ForesightQdrantConverter(BaseQdrantConverter[ForesightCollection]):
    """Converts MongoDB ``v1_foresight_records`` documents to Qdrant point payloads."""

    @classmethod
    def _parse_time_field(
        cls,
        time_value: Optional[Union[datetime, str, int, float]],
        field_name: str,
        doc_id: Optional[str],
    ) -> int:
        """
        Parse a time field to epoch milliseconds.

        Accepts ``datetime``, ISO-8601 strings, numeric epoch seconds, or
        numeric epoch milliseconds — the magnitude guard distinguishes the
        two numeric units (values above 1e10 are treated as already-ms,
        otherwise multiplied by 1000). This intentionally diverges from the
        Milvus template, which always multiplies numeric inputs by 1000 and
        thus would corrupt already-ms inputs.
        """
        # Explicit ``is None`` so a legitimate epoch 0 / datetime(1970-01-01)
        # is not silently dropped as "missing".
        if time_value is None:
            return 0

        try:
            if isinstance(time_value, datetime):
                return int(time_value.timestamp() * 1000)
            if isinstance(time_value, str):
                dt = datetime.fromisoformat(time_value.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            if isinstance(time_value, (int, float)):
                # Magnitude guard: 1e10 epoch-seconds ~= year 2286, so any
                # numeric > 1e10 is already in milliseconds.
                value_ms = time_value if time_value > 1e10 else time_value * 1000
                return int(value_ms)
        except Exception as e:
            logger.warning(
                "Failed to parse %s (doc_id=%s): %s, error: %s",
                field_name, doc_id, time_value, e,
            )

        return 0

    @classmethod
    def from_mongo(cls, source_doc: MongoForesightRecord) -> qmodels.PointStruct:
        """
        Build a ``PointStruct`` from a MongoDB foresight-record document.

        Raises:
            ValueError: when ``source_doc`` is ``None``.
            Exception: on any conversion failure (logged + re-raised).
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be None")
        if source_doc.id is None:
            raise ValueError("ForesightRecord.id must not be None")

        try:
            start_time = cls._parse_time_field(
                source_doc.start_time, "start_time", source_doc.id
            )
            end_time = cls._parse_time_field(
                source_doc.end_time, "end_time", source_doc.id
            )

            search_content = cls._build_search_content(source_doc)

            payload = {
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "session_id": source_doc.session_id or "",
                "participants": source_doc.participants or [],
                "sender_ids": getattr(source_doc, "sender_ids", []) or [],
                "type": getattr(source_doc, "type", None) or "",
                "start_time": start_time,
                "end_time": end_time,
                "duration_days": (
                    source_doc.duration_days if source_doc.duration_days else 0
                ),
                # ``content`` is intentionally passed through verbatim (incl.
                # ``None``) — parity with the Milvus template. Downstream
                # search code distinguishes "absent content" from "empty
                # content" via the ``None`` sentinel.
                "content": source_doc.content,
                "evidence": source_doc.evidence or "",
                "search_content": search_content,
                "parent_type": source_doc.parent_type or "",
                "parent_id": (
                    str(source_doc.parent_id) if source_doc.parent_id else ""
                ),
                # Mongo back-reference (see episodic_memory converter).
                "mongo_id": str(source_doc.id),
            }

            vector = source_doc.vector if source_doc.vector else None
            if not vector:
                raise ValueError(
                    f"Vector is required for ForesightRecord {source_doc.id} "
                    "but was not populated"
                )

            return qmodels.PointStruct(
                id=mongo_id_to_qdrant_id(source_doc.id),
                vector=vector,
                payload=payload,
            )

        except Exception as e:
            logger.exception(
                "Failed to convert MongoDB foresight document to Qdrant point: %s", e
            )
            raise

    @staticmethod
    def _build_search_content(source_doc: MongoForesightRecord) -> str:
        """Build search content JSON-string from content + evidence fields."""
        text_content: List[str] = []
        if source_doc.content:
            text_content.append(source_doc.content)
        if source_doc.evidence:
            text_content.append(source_doc.evidence)
        return json.dumps(text_content, ensure_ascii=False)
