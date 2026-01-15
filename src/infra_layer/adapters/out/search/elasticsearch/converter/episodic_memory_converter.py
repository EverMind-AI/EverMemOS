"""
EpisodicMemory ES Converter

Responsible for converting EpisodicMemory documents from MongoDB to EpisodicMemoryDoc documents in Elasticsearch.
"""

from typing import List
import jieba
from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger

# EpisodicMemory type no longer needs to be imported, as parameter types have been simplified to Any
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
    EpisodicMemoryDoc,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory as MongoEpisodicMemory,
)

logger = get_logger(__name__)


class EpisodicMemoryConverter(BaseEsConverter[EpisodicMemoryDoc]):
    """
    EpisodicMemory Converter

    Converts EpisodicMemory documents from MongoDB to EpisodicMemoryDoc documents in Elasticsearch.
    Mainly handles field mapping, search content construction, and data format conversion.
    ES document type is automatically obtained from the generic BaseEsConverter[EpisodicMemoryDoc].
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEpisodicMemory) -> EpisodicMemoryDoc:
        """
        Convert from MongoDB EpisodicMemory document to ES EpisodicMemoryDoc instance

        Usage scenarios:
        - During ES index rebuilding, convert MongoDB documents to ES documents
        - During data synchronization, ensure MongoDB data is correctly mapped to ES fields
        - Handle field mapping and data format conversion

        Args:
            source_doc: Instance of MongoDB's EpisodicMemory document

        Returns:
            EpisodicMemoryDoc: ES document instance, ready for indexing

        Raises:
            Exception: Throws an exception if an error occurs during conversion
        """
        # Basic validation
        if source_doc is None:
            raise ValueError("MongoDB document cannot be empty")

        try:
            # Build search content list for BM25 retrieval
            search_content = cls._build_search_content(source_doc)

            # Create ES document instance
            es_doc = EpisodicMemoryDoc(
                # Basic identifier fields
                event_id=(
                    str(source_doc.id)
                    if hasattr(source_doc, 'id') and source_doc.id
                    else ""
                ),
                user_id=source_doc.user_id,
                user_name=getattr(source_doc, 'user_name', None),
                # Timestamp fields
                timestamp=source_doc.timestamp,
                # Core content fields
                title=getattr(
                    source_doc, 'subject', None
                ),  # Map MongoDB's subject to ES's title
                episode=source_doc.episode,
                search_content=search_content,  # Core field for BM25 search
                summary=getattr(source_doc, 'summary', None),
                # Category and tag fields
                group_id=getattr(source_doc, 'group_id', None),
                participants=getattr(source_doc, 'participants', None),
                type=getattr(source_doc, 'type', None),
                keywords=getattr(source_doc, 'keywords', None),
                linked_entities=getattr(source_doc, 'linked_entities', None),
                # MongoDB-specific fields
                subject=getattr(source_doc, 'subject', None),
                memcell_event_id_list=getattr(
                    source_doc, 'memcell_event_id_list', None
                ),
                # Parent info
                parent_type=getattr(source_doc, 'parent_type', None),
                parent_id=getattr(source_doc, 'parent_id', None),
                # Extension fields
                extend=getattr(source_doc, 'extend', None),
                # Audit fields
                created_at=getattr(source_doc, 'created_at', None),
                updated_at=getattr(source_doc, 'updated_at', None),
            )

            return es_doc

        except Exception as e:
            logger.error("Failed to convert MongoDB document to ES document: %s", e)
            raise

    @classmethod
    def _build_search_content(cls, source_doc: MongoEpisodicMemory) -> List[str]:
        """
        Build search content list

        Combines multiple text fields from the MongoDB document and processes them with jieba word segmentation,
        generating a list of search content for BM25 retrieval.

        Args:
            source_doc: Instance of MongoDB's EpisodicMemory document

        Returns:
            List[str]: List of search content after jieba word segmentation
        """
        text_content = []

        # Collect all text content - including subject, summary, episode
        if hasattr(source_doc, 'subject') and source_doc.subject:
            text_content.append(source_doc.subject)

        if hasattr(source_doc, 'summary') and source_doc.summary:
            text_content.append(source_doc.summary)

        if hasattr(source_doc, 'episode') and source_doc.episode:
            text_content.append(source_doc.episode)

        # Combine all text content and apply jieba word segmentation
        combined_text = ' '.join(text_content)
        search_content = list(jieba.cut(combined_text))

        # Filter out empty strings
        query_words = filter_stopwords(search_content, min_length=2)

        search_content = [word.strip() for word in query_words if word.strip()]

        return search_content

    @classmethod
    def from_memory(cls, episodic_memory) -> EpisodicMemoryDoc:
        """
        Convert from Memory object to ES EpisodicMemoryDoc instance

        !!!!!!!!!!!! Remove this later !!!!!!!!!!!!!!!
        Going forward, ES will be derived entirely from MongoDB, following the single source of truth principle.

        Specifically used for handling Memory objects obtained from memory_manager,
        including jieba word segmentation and field mapping logic.

        Args:
            episodic_memory: Instance of Memory object

        Returns:
            EpisodicMemoryDoc: ES document instance, ready for indexing

        Raises:
            Exception: Throws an exception if an error occurs during conversion
        """
        raise NotImplementedError(
            "The from_memory method is no longer used. Please use the from_mongo method to convert from MongoDB document"
        )
