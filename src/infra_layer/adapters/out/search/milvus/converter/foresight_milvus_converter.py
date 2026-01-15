"""
Foresight Milvus Converter

Responsible for converting MongoDB foresight documents into Milvus Collection entities, supporting both individual and group use cases.
"""

from typing import Dict, Any
import json
from datetime import datetime

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.foresight_collection import (
    ForesightCollection,
)
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord as MongoForesightRecord,
)

logger = get_logger(__name__)


class ForesightMilvusConverter(BaseMilvusConverter[ForesightCollection]):
    """
    Foresight Milvus Converter

    Converts MongoDB foresight documents into Milvus Collection entities.
    Uses an independent ForesightCollection, supporting both individual and group foresights.
    """

    @classmethod
    def _parse_time_field(cls, time_value, field_name: str, doc_id) -> int:
        """Parse time field, return 0 and log warning on failure"""
        if not time_value:
            return 0

        try:
            if isinstance(time_value, datetime):
                return int(time_value.timestamp())
            elif isinstance(time_value, str):
                dt = datetime.fromisoformat(time_value.replace('Z', '+00:00'))
                return int(dt.timestamp())
            elif isinstance(time_value, (int, float)):
                return int(time_value)
        except Exception as e:
            logger.warning(
                f"Failed to parse {field_name} (doc_id={doc_id}): {time_value}, error: {e}"
            )

        return 0

    @classmethod
    def _parse_time_field(cls, time_value, field_name: str, doc_id) -> int:
        """Parse time field, return 0 and log warning on failure"""
        if not time_value:
            return 0

        try:
            if isinstance(time_value, datetime):
                return int(time_value.timestamp())
            elif isinstance(time_value, str):
                dt = datetime.fromisoformat(time_value.replace('Z', '+00:00'))
                return int(dt.timestamp())
            elif isinstance(time_value, (int, float)):
                return int(time_value)
        except Exception as e:
            logger.warning(
                f"Failed to parse {field_name} (doc_id={doc_id}): {time_value}, error: {e}"
            )

        return 0

    @classmethod
    def from_mongo(cls, source_doc: MongoForesightRecord) -> Dict[str, Any]:
        """
        Convert from MongoDB foresight document to Milvus Collection entity

        Args:
            source_doc: MongoDB foresight document instance

        Returns:
            Dict[str, Any]: Milvus entity dictionary, ready for insertion
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be None")

        try:
            # Parse time fields
            start_time = cls._parse_time_field(
                source_doc.start_time, "start_time", source_doc.id
            )
            end_time = cls._parse_time_field(
                source_doc.end_time, "end_time", source_doc.id
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
                # Time fields
                "start_time": start_time,
                "end_time": end_time,
                "duration_days": (
                    source_doc.duration_days if source_doc.duration_days else 0
                ),
                # Core content fields
                "content": source_doc.content,
                "evidence": source_doc.evidence or "",
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
                "Failed to convert MongoDB foresight document to Milvus entity: %s", e
            )
            raise

    @classmethod
    def _build_detail(cls, source_doc: MongoForesightRecord) -> Dict[str, Any]:
        """Build detailed information dictionary"""
        detail = {"vector_model": source_doc.vector_model, "extend": source_doc.extend}

        # Filter out None values
        return {k: v for k, v in detail.items() if v is not None}

    @staticmethod
    def _build_search_content(source_doc: MongoForesightRecord) -> str:
        """Build search content (JSON list format)"""
        text_content = []

        # Main content
        if source_doc.content:
            text_content.append(source_doc.content)

        # Add evidence to improve retrieval capability
        if source_doc.evidence:
            text_content.append(source_doc.evidence)

        return json.dumps(text_content, ensure_ascii=False)
