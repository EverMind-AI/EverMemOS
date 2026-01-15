"""
Event Log Milvus Repository

A dedicated repository class for event logs based on BaseMilvusRepository, providing efficient vector storage and retrieval capabilities.
Key features include vector storage, similarity search, filtered queries, and document management.
Supports both personal and group event logs.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import json
from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.event_log_collection import (
    EventLogCollection,
)
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone
from core.di.decorators import repository

logger = get_logger(__name__)


@repository("event_log_milvus_repository", primary=False)
class EventLogMilvusRepository(BaseMilvusRepository[EventLogCollection]):
    """
    Event Log Milvus Repository

    A dedicated repository class based on BaseMilvusRepository, providing:
    - Efficient vector storage and retrieval
    - Similarity search and filtering capabilities
    - Document creation and management
    - Vector index management

    Supports both personal and group event logs.
    """

    def __init__(self):
        """Initialize the event log repository"""
        super().__init__(EventLogCollection)

    # ==================== Document Creation and Management ====================

    async def create_and_save_event_log(
        self,
        id: str,
        user_id: Optional[str],
        atomic_fact: str,
        parent_id: str,
        parent_type: str,
        timestamp: datetime,
        vector: List[float],
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        search_content: Optional[List[str]] = None,
        extend: Optional[Dict[str, Any]] = None,
        vector_model: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create and save an event log document

        Args:
            id: Unique identifier for the event log
            user_id: User ID (required)
            atomic_fact: Atomic fact content (required)
            parent_id: Parent memory ID (required)
            parent_type: Parent memory type (memcell/episode)
            timestamp: Event occurrence time (required)
            vector: Text vector (required, dimension must be VECTORIZE_DIMENSIONS)
            group_id: Group ID
            participants: List of related participants
            event_type: Event type (e.g., Conversation, Email, etc.)
            search_content: List of searchable content
            extend: Extension fields
            vector_model: Vectorization model
            created_at: Creation time
            updated_at: Update time

        Returns:
            Information of the saved document
        """
        try:
            # Set default timestamps
            now = get_now_with_timezone()
            if created_at is None:
                created_at = now
            if updated_at is None:
                updated_at = now

            # Build search content
            if search_content is None:
                search_content = [atomic_fact]

            # Prepare metadata
            metadata = {"vector_model": vector_model or "", "extend": extend or {}}

            # Prepare entity data
            entity = {
                "id": id,
                "vector": vector,
                "user_id": user_id or "",
                "group_id": group_id or "",
                "participants": participants or [],
                "parent_type": parent_type,
                "parent_id": parent_id,
                "event_type": event_type,
                "timestamp": int(timestamp.timestamp()),
                "atomic_fact": atomic_fact,
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "created_at": int(created_at.timestamp()),
                "updated_at": int(updated_at.timestamp()),
            }

            # Insert data
            await self.insert(entity)

            logger.debug(
                "✅ Successfully created event log document: id=%s, user_id=%s",
                id,
                user_id,
            )

            return {
                "id": id,
                "user_id": user_id,
                "atomic_fact": atomic_fact,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "timestamp": timestamp,
                "search_content": search_content,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(
                "❌ Failed to create event log document: id=%s, error=%s", id, e
            )
            raise

    # ==================== Search Functionality ====================

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
        participant_user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search

        Args:
            query_vector: Query vector
            user_id: User ID filter
            group_id: Group ID filter
            parent_type: Parent type filter (e.g., "memcell", "episode")
            parent_id: Parent memory ID filter
            event_type: Event type filter
            start_time: Start timestamp filter
            end_time: End timestamp filter
            limit: Number of results to return
            score_threshold: Similarity score threshold
            participant_user_id: For group retrieval, additionally require this user to be in participants

        Returns:
            List of search results
        """
        try:
            # Build filter expression
            filter_expr = []

            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL:
                if user_id:
                    filter_expr.append(f'user_id == "{user_id}"')
                else:
                    # Explicitly filter for null or empty
                    filter_expr.append('user_id == ""')

            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL:
                if group_id:
                    filter_expr.append(f'group_id == "{group_id}"')
                else:
                    # Explicitly filter for null or empty
                    filter_expr.append('group_id == ""')

            if participant_user_id:
                filter_expr.append(
                    f'array_contains(participants, "{participant_user_id}")'
                )
            if parent_type:
                filter_expr.append(f'parent_type == "{parent_type}"')
            if parent_id:
                filter_expr.append(f'parent_id == "{parent_id}"')
            if event_type:
                filter_expr.append(f'event_type == "{event_type}"')
            if start_time:
                filter_expr.append(f"timestamp >= {int(start_time.timestamp())}")
            if end_time:
                filter_expr.append(f"timestamp <= {int(end_time.timestamp())}")

            filter_str = " and ".join(filter_expr) if filter_expr else None

            similarity_threshold: Optional[float] = (
                radius if radius is not None else None
            )

            # Execute search
            # Dynamically adjust ef parameter: must be >= limit, typically set to 1.5-2 times limit
            ef_value = max(128, limit * 2)  # Ensure ef >= limit, minimum 128
            search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}

            # Do not set radius parameter!
            # Milvus radius is a similarity lower bound; setting -1.0 may cause issues
            # We use post-filtering to control similarity threshold
            if radius is not None and radius > -1.0:
                search_params["params"]["radius"] = radius

            logger.info(
                f"Milvus search parameters: limit={limit}, "
                f"radius={search_params['params'].get('radius', 'None')}, "
                f"filter_str={filter_str}"
            )

            results = await self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filter_str,
                output_fields=self.all_output_fields,
            )

            # Process results
            search_results = []
            raw_hit_count = sum(len(hits) for hits in results)
            logger.info(
                f"Milvus raw response: {raw_hit_count} results, "
                f"limit={limit}, filter_str={filter_str}, "
            )

            for hits in results:
                for hit in hits:
                    threshold = (
                        similarity_threshold
                        if similarity_threshold is not None
                        else score_threshold
                    )
                    keep = hit.score >= threshold

                    if keep:
                        # Parse metadata
                        metadata_json = hit.entity.get("metadata", "{}")
                        metadata = json.loads(metadata_json) if metadata_json else {}

                        # Parse search_content (unified as JSON array format)
                        search_content_raw = hit.entity.get("search_content", "[]")
                        search_content = (
                            json.loads(search_content_raw) if search_content_raw else []
                        )

                        # Build result
                        result = {
                            "id": hit.entity.get("id"),
                            "score": float(hit.score),
                            "user_id": hit.entity.get("user_id"),
                            "group_id": hit.entity.get("group_id"),
                            "parent_type": hit.entity.get("parent_type"),
                            "parent_id": hit.entity.get("parent_id"),
                            "event_type": hit.entity.get("event_type"),
                            "timestamp": datetime.fromtimestamp(
                                hit.entity.get("timestamp", 0)
                            ),
                            "atomic_fact": hit.entity.get("atomic_fact"),
                            "search_content": search_content,
                            "metadata": metadata,
                        }
                        search_results.append(result)

            logger.debug(
                "✅ Vector search successful: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("❌ Vector search failed: %s", e)
            raise

    # ==================== Deletion Functionality ====================

    async def delete_by_id(self, log_id: str) -> bool:
        """
        Delete event log document by log_id

        Args:
            log_id: Unique identifier of the event log

        Returns:
            True if deletion succeeds, otherwise False
        """
        try:
            success = await super().delete_by_id(log_id)
            if success:
                logger.debug(
                    "✅ Successfully deleted event log by log_id: log_id=%s", log_id
                )
            return success
        except Exception as e:
            logger.error(
                "❌ Failed to delete event log by log_id: log_id=%s, error=%s",
                log_id,
                e,
            )
            return False

    async def delete_by_parent_id(
        self, parent_id: str, parent_type: Optional[str] = None
    ) -> int:
        """
        Delete all associated event logs by parent memory ID and optionally parent type

        Args:
            parent_id: Parent memory ID
            parent_type: Optional parent type filter (e.g., "memcell", "episode")

        Returns:
            Number of deleted documents
        """
        try:
            expr = f'parent_id == "{parent_id}"'
            if parent_type is not None:
                expr += f' and parent_type == "{parent_type}"'

            # First query the number of documents to delete
            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            if delete_count > 0:
                # Perform deletion
                await self.collection.delete(expr)

            logger.debug(
                "✅ Successfully deleted event logs by parent_id: %s (type=%s), deleted %d records",
                parent_id,
                parent_type,
                delete_count,
            )
            return delete_count

        except Exception as e:
            logger.error("❌ Failed to delete event logs by parent_id: %s", e)
            raise

    async def delete_by_filters(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Batch delete event log documents based on filter conditions

        Args:
            user_id: User ID filter
            group_id: Group ID filter
            start_time: Start time
            end_time: End time

        Returns:
            Number of deleted documents
        """
        try:
            # Build filter expression
            filter_expr = []
            # Handle user_id filter: MAGIC_ALL means no filter
            if user_id != MAGIC_ALL and user_id is not None:
                if user_id:  # Non-empty string: personal memory
                    # Check both user_id field and participants array
                    user_filter = f'(user_id == "{user_id}" or array_contains(participants, "{user_id}"))'
                    filter_expr.append(user_filter)
                else:  # Empty string: group memory
                    filter_expr.append('user_id == ""')
            # Handle group_id filter: MAGIC_ALL means no filter
            if group_id != MAGIC_ALL and group_id:
                filter_expr.append(f'group_id == "{group_id}"')
            if start_time:
                filter_expr.append(f"timestamp >= {int(start_time.timestamp())}")
            if end_time:
                filter_expr.append(f"timestamp <= {int(end_time.timestamp())}")

            if not filter_expr:
                raise ValueError("At least one filter condition must be provided")

            expr = " and ".join(filter_expr)

            # First query the number of documents to delete
            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            # Perform deletion
            await self.collection.delete(expr)

            logger.debug(
                "✅ Successfully batch deleted event logs by filters: deleted %d records",
                delete_count,
            )
            return delete_count

        except Exception as e:
            logger.error("❌ Failed to batch delete event logs by filters: %s", e)
            raise
