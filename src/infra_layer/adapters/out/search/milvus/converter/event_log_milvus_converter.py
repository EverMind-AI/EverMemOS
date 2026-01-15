"""
Event Log Milvus Converter

Responsible for converting MongoDB EventLog documents into Milvus Collection entities, supporting both individual and group scenarios.
"""

from typing import Dict, Any
import json

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.event_log_collection import (
    EventLogCollection,
)
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord as MongoEventLogRecord,
)
from api_specs.memory_types import RawDataType

logger = get_logger(__name__)


class EventLogMilvusConverter(BaseMilvusConverter[EventLogCollection]):
    """
    Event Log Milvus Converter

    Converts MongoDB EventLog documents into Milvus Collection entities.
    Uses an independent EventLogCollection, supporting both individual and group event logs.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEventLogRecord) -> Dict[str, Any]:
        """
        Convert MongoDB EventLog document to Milvus Collection entity

        Args:
            source_doc: Instance of MongoDB EventLog document

        Returns:
            Dict[str, Any]: Milvus entity dictionary, ready for insertion
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be None")

        try:
            # Convert timestamp
            timestamp = (
                int(source_doc.timestamp.timestamp()) if source_doc.timestamp else 0
            )

            # Build search content
            search_content = cls._build_search_content(source_doc)

            # Create Milvus entity dictionary
            milvus_entity = {
                # Basic identifier fields
                "id": str(source_doc.id),  # Use Beanie's id attribute
                "user_id": source_doc.user_id or "",
                "group_id": source_doc.group_id or "",
                "participants": (
                    source_doc.participants if source_doc.participants else []
                ),
                "parent_type": source_doc.parent_type,
                "parent_id": source_doc.parent_id,
                # Event type and time fields
                "event_type": source_doc.event_type or RawDataType.CONVERSATION.value,
                "timestamp": timestamp,
                # Core content fields
                "atomic_fact": source_doc.atomic_fact,
                "search_content": search_content,
                # Detailed information in JSON
                "metadata": json.dumps(
                    cls._build_detail(source_doc), ensure_ascii=False
                ),
                # Audit fields
                "created_at": (
                    int(source_doc.created_at.timestamp())
                    if source_doc.created_at
                    else 0
                ),
                "updated_at": (
                    int(source_doc.updated_at.timestamp())
                    if source_doc.updated_at
                    else 0
                ),
                # Vector field
                "vector": source_doc.vector if source_doc.vector else [],
            }

            return milvus_entity

        except Exception as e:
            logger.error(
                "Failed to convert MongoDB EventLog document to Milvus entity: %s", e
            )
            raise

    @classmethod
    def _build_detail(cls, source_doc: MongoEventLogRecord) -> Dict[str, Any]:
        """Build detailed information dictionary"""
        detail = {"vector_model": source_doc.vector_model, "extend": source_doc.extend}

        # Filter out None values
        return {k: v for k, v in detail.items() if v is not None}

    @staticmethod
    def _build_search_content(source_doc: MongoEventLogRecord) -> str:
        """Build search content (JSON list format)"""
        text_content = []

        if source_doc.atomic_fact:
            text_content.append(source_doc.atomic_fact)

        return json.dumps(text_content, ensure_ascii=False)
