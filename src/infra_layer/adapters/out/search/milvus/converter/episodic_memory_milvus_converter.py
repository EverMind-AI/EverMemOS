"""
Episodic Memory Milvus Converter

Responsible for converting MongoDB's EpisodicMemory documents into Milvus Collection entities.
Mainly handles field mapping, vector construction, and data format conversion.
"""

from typing import Dict, Any
import json

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory as MongoEpisodicMemory,
)

logger = get_logger(__name__)


class EpisodicMemoryMilvusConverter(BaseMilvusConverter[EpisodicMemoryCollection]):
    """
    EpisodicMemory Milvus Converter

    Converts MongoDB EpisodicMemory documents into Milvus Collection entities.
    Mainly handles field mapping, vector building, and data format conversion.
    Milvus Collection type is automatically obtained from the generic BaseMilvusConverter[EpisodicMemoryCollection].
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEpisodicMemory) -> Dict[str, Any]:
        """
        Convert from MongoDB EpisodicMemory document to Milvus Collection entity

        Use cases:
        - During Milvus index rebuilding, convert MongoDB documents into Milvus entities
        - During data synchronization, ensure MongoDB data is correctly mapped to Milvus fields
        - Handle field mapping and data format conversion

        Args:
            source_doc: MongoDB EpisodicMemory document instance

        Returns:
            Dict[str, Any]: Milvus entity dictionary, can be directly used for insertion

        Raises:
            Exception: Raises an exception when an error occurs during conversion
        """
        # Basic validation
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            # Convert timestamp to integer
            timestamp = (
                int(source_doc.timestamp.timestamp()) if source_doc.timestamp else 0
            )

            # Create Milvus entity dictionary
            milvus_entity = {
                # Basic identifier fields
                "id": str(source_doc.id),  # Use Beanie's id attribute
                "user_id": source_doc.user_id or "",  # Convert None to empty string
                "group_id": getattr(source_doc, 'group_id', ""),
                "participants": getattr(
                    source_doc, 'participants', []
                ),  # Add participants
                # Time fields - convert to Unix timestamp
                "timestamp": timestamp,
                # Core content fields
                "episode": source_doc.episode,
                "search_content": cls._build_search_content(source_doc),
                # Classification fields
                "event_type": (
                    str(source_doc.type)
                    if hasattr(source_doc, 'type') and source_doc.type
                    else ""
                ),
                # Metadata JSON (detailed information)
                "metadata": json.dumps(
                    cls._build_detail(source_doc), ensure_ascii=False
                ),
                # Parent info
                "parent_type": getattr(source_doc, 'parent_type', None) or "",
                "parent_id": getattr(source_doc, 'parent_id', None) or "",
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
                # Vector field - needs to be set externally
                "vector": (
                    source_doc.vector
                    if hasattr(source_doc, 'vector') and source_doc.vector
                    else []
                ),
            }

            return milvus_entity

        except Exception as e:
            logger.error("Failed to convert MongoDB document to Milvus entity: %s", e)
            raise

    @classmethod
    def _build_detail(cls, source_doc: MongoEpisodicMemory) -> Dict[str, Any]:
        """
        Build detailed information dictionary

        Consolidate data that is not suitable for direct storage in Milvus fields into the detail JSON.
        These data are typically not used for retrieval but may need to be displayed when retrieving results.

        Args:
            source_doc: MongoDB EpisodicMemory document instance

        Returns:
            Dict[str, Any]: Detailed information dictionary
        """
        detail = {
            # User information
            "user_name": getattr(source_doc, 'user_name', None),
            # Content related
            "title": getattr(source_doc, 'subject', None),
            "summary": getattr(source_doc, 'summary', None),
            # Classification and tags
            "participants": getattr(source_doc, 'participants', None),
            "keywords": getattr(source_doc, 'keywords', None),
            "linked_entities": getattr(source_doc, 'linked_entities', None),
            # MongoDB specific fields
            "subject": getattr(source_doc, 'subject', None),
            "memcell_event_id_list": getattr(source_doc, 'memcell_event_id_list', None),
            # Extension fields
            "extend": getattr(source_doc, 'extend', None),
        }

        # Filter out None values
        return {k: v for k, v in detail.items() if v is not None}

    @staticmethod
    def _build_search_content(source_doc: MongoEpisodicMemory) -> str:
        """
        Build search content

        Combine key text content from the document into a search content list, return as JSON string.

        Args:
            source_doc: MongoDB EpisodicMemory document instance

        Returns:
            str: Search content JSON string (list format)
        """
        text_content = []

        # Collect all text content (by priority: subject -> summary -> content)
        if hasattr(source_doc, 'subject') and source_doc.subject:
            text_content.append(source_doc.subject)

        if hasattr(source_doc, 'summary') and source_doc.summary:
            text_content.append(source_doc.summary)

        if hasattr(source_doc, 'episode') and source_doc.episode:
            # episode might be very long, only take first 500 characters
            text_content.append(source_doc.episode)

        # Return JSON string list format, keep consistent with MemCell synchronization logic
        return json.dumps(text_content, ensure_ascii=False)
