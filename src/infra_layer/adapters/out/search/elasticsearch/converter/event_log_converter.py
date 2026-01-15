"""
Event Log ES Converter

Converts MongoDB EventLog to Elasticsearch EventLogDoc document.
Supports both personal and group event logs.
"""

from typing import List
import jieba

from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.event_log import EventLogDoc
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord as MongoEventLogRecord,
)
from api_specs.memory_types import RawDataType

logger = get_logger(__name__)


class EventLogConverter(BaseEsConverter[EventLogDoc]):
    """
    Event Log ES Converter

    Converts MongoDB event log documents to Elasticsearch EventLogDoc documents.
    Supports both personal and group event logs.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEventLogRecord) -> EventLogDoc:
        """
        Convert from MongoDB event log document to ES EventLogDoc document

        Args:
            source_doc: Instance of MongoDB event log document

        Returns:
            EventLogDoc: Instance of ES document
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            # Build search content list for BM25 retrieval
            search_content = cls._build_search_content(source_doc)

            # Create ES document instance
            # Pass id via meta parameter to ensure idempotency (MongoDB _id -> ES _id)
            es_doc = EventLogDoc(
                meta={'id': str(source_doc.id)},
                user_id=source_doc.user_id,
                user_name=source_doc.user_name or "",
                # Timestamp field
                timestamp=source_doc.timestamp,
                # Core content field
                search_content=search_content,
                atomic_fact=source_doc.atomic_fact,
                # Classification and tag fields
                group_id=source_doc.group_id,
                group_name=source_doc.group_name or "",
                participants=source_doc.participants,
                type=source_doc.event_type or RawDataType.CONVERSATION.value,
                # Parent info
                parent_type=source_doc.parent_type,
                parent_id=source_doc.parent_id,
                # Extension fields
                extend={
                    "vector_model": source_doc.vector_model,
                    **(source_doc.extend or {}),
                },
                # Audit fields
                created_at=source_doc.created_at,
                updated_at=source_doc.updated_at,
            )

            return es_doc

        except Exception as e:
            logger.error(
                "Failed to convert MongoDB event log document to ES document: %s", e
            )
            raise

    @classmethod
    def _build_search_content(cls, source_doc: MongoEventLogRecord) -> List[str]:
        """
        Build search content list

        Perform word segmentation on Chinese text, filter stop words, and generate keyword list for BM25 retrieval.
        """
        search_content = []

        # Segment atomic_fact
        if source_doc.atomic_fact:
            words = jieba.lcut(source_doc.atomic_fact)
            # Use min_length=2 to retain meaningful words and avoid over-filtering
            words = filter_stopwords(words, min_length=2)
            search_content.extend(words)

        # Deduplicate while preserving order
        seen = set()
        unique_content = []
        for word in search_content:
            if word not in seen and word.strip():
                seen.add(word)
                unique_content.append(word)

        # If empty after filtering, use original text as fallback
        if not unique_content and source_doc.atomic_fact:
            return [source_doc.atomic_fact]

        return unique_content if unique_content else [""]
