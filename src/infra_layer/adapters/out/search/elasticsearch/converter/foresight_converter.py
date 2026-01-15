"""
Foresight ES Converter

Responsible for converting MongoDB foresight documents into Elasticsearch ForesightDoc documents.
Supports both individual and group foresights.
"""

from typing import List
import jieba

from common_utils.datetime_utils import get_now_with_timezone
from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.foresight import ForesightDoc
from infra_layer.adapters.out.persistence.document.memory.foresight_record import (
    ForesightRecord as MongoForesightRecord,
)
from datetime import datetime

logger = get_logger(__name__)


class ForesightConverter(BaseEsConverter[ForesightDoc]):
    """
    Foresight ES Converter

    Converts MongoDB foresight documents into Elasticsearch ForesightDoc documents.
    Supports both individual and group foresights.
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoForesightRecord) -> ForesightDoc:
        """
        Convert from MongoDB foresight document to ES ForesightDoc document

        Args:
            source_doc: MongoDB foresight document instance

        Returns:
            ForesightDoc: ES document instance
        """
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            # Build search content list for BM25 retrieval
            search_content = cls._build_search_content(source_doc)

            # Parse timestamp
            timestamp = None
            if source_doc.start_time:
                if isinstance(source_doc.start_time, str):
                    timestamp = datetime.fromisoformat(
                        source_doc.start_time.replace('Z', '+00:00')
                    )
                elif isinstance(source_doc.start_time, datetime):
                    timestamp = source_doc.start_time

            if not timestamp:
                timestamp = source_doc.created_at or get_now_with_timezone()

            # Create ES document instance
            # Pass id via meta parameter to ensure idempotency (MongoDB _id -> ES _id)
            es_doc = ForesightDoc(
                meta={'id': str(source_doc.id)},
                user_id=source_doc.user_id,
                user_name=source_doc.user_name or "",
                # Timestamp field
                timestamp=timestamp,
                # Core content fields
                foresight=source_doc.content,
                evidence=source_doc.evidence or "",
                search_content=search_content,
                # Categorization and tagging fields
                group_id=source_doc.group_id,
                group_name=source_doc.group_name or "",
                participants=source_doc.participants,
                type="Conversation",
                # Parent info
                parent_type=source_doc.parent_type,
                parent_id=source_doc.parent_id,
                # Extension fields
                extend={
                    "start_time": source_doc.start_time,
                    "end_time": source_doc.end_time,
                    "duration_days": source_doc.duration_days,
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
                "Failed to convert MongoDB foresight document to ES document: %s", e
            )
            raise

    @classmethod
    def _build_search_content(cls, source_doc: MongoForesightRecord) -> List[str]:
        """
        Build search content list

        Perform Chinese text tokenization and filter out stop words to generate keyword list for BM25 retrieval.
        """
        search_content = []

        # Tokenize content
        if source_doc.content:
            words = jieba.lcut(source_doc.content)
            words = filter_stopwords(words)
            search_content.extend(words)

        # # Tokenize evidence
        # if source_doc.evidence:
        #     words = jieba.lcut(source_doc.evidence)
        #     words = filter_stopwords(words)
        #     search_content.extend(words)

        # Deduplicate while preserving order
        seen = set()
        unique_content = []
        for word in search_content:
            if word not in seen and word.strip():
                seen.add(word)
                unique_content.append(word)

        return unique_content if unique_content else [""]
