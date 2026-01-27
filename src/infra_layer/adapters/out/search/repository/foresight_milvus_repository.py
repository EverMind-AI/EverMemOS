"""
Foresight Milvus Repository

Specialized repository class based on BaseMilvusRepository, providing efficient vector storage and retrieval capabilities.
Main features include vector storage, similarity search, filtered queries, and document management.
Supports both personal foresight and group foresight.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import json
from core.oxm.milvus.base_repository import BaseMilvusRepository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.milvus.memory.foresight_collection import (
    ForesightCollection,
)
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone
from core.di.decorators import repository


logger = get_logger(__name__)

# Milvus retrieval configuration (None means radius filtering is disabled)
MILVUS_SIMILARITY_RADIUS = None  # COSINE similarity threshold, optional range [-1, 1]


@repository("foresight_milvus_repository", primary=False)
class ForesightMilvusRepository(BaseMilvusRepository[ForesightCollection]):
    """
    Foresight Milvus Repository

    Specialized repository class based on BaseMilvusRepository, providing:
    - Efficient vector storage and retrieval
    - Similarity search and filtering capabilities
    - Document creation and management
    - Vector index management

    Supports both personal foresight and group foresight.
    """

    def __init__(self):
        """Initialize foresight repository"""
        super().__init__(ForesightCollection)

    # ==================== Document Creation and Management ====================
    # TODO: add username
    async def create_and_save_foresight_mem(
        self,
        id: str,
        user_id: Optional[str],
        content: str,
        parent_id: str,
        parent_type: str,
        vector: List[float],
        group_id: Optional[str] = None,
        event_type: Optional[str] = None,
        participants: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_days: Optional[int] = None,
        evidence: Optional[str] = None,
        search_content: Optional[List[str]] = None,
        extend: Optional[Dict[str, Any]] = None,
        vector_model: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create and save personal foresight document

        Args:
            id: Unique identifier for foresight
            user_id: User ID (required)
            content: Foresight content (required)
            parent_id: Parent memory ID (required)
            parent_type: Parent memory type (memcell/episode)
            vector: Text vector (required)
            group_id: Group ID
            participants: List of related participants
            start_time: Foresight start time
            end_time: Foresight end time
            duration_days: Duration in days
            evidence: Evidence supporting this foresight
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
                search_content = [content]
                if evidence:
                    search_content.append(evidence)

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
                "start_time": int(start_time.timestamp()) if start_time else 0,
                "end_time": int(end_time.timestamp()) if end_time else 0,
                "duration_days": duration_days or 0,
                "content": content,
                "evidence": evidence or "",
                "event_type": event_type,
                "search_content": json.dumps(search_content, ensure_ascii=False),
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "created_at": int(created_at.timestamp()),
                "updated_at": int(updated_at.timestamp()),
            }

            # Insert data
            await self.insert(entity)

            logger.debug(
                "✅ Created personal foresight document successfully: memory_id=%s, user_id=%s",
                id,
                user_id,
            )

            return {
                "id": id,
                "user_id": user_id,
                "content": content,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "search_content": search_content,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(
                "❌ Failed to create personal foresight document: id=%s, error=%s",
                id,
                e,
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
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
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
            start_time: Foresight start time filter
            end_time: Foresight end time filter
            current_time: Current time, used to filter foresights within validity period
            limit: Number of results to return
            score_threshold: Similarity threshold
            radius: COSINE similarity threshold (optional, defaults to MILVUS_SIMILARITY_RADIUS)
            participant_user_id: When retrieving group data, additionally require this user to be in participants

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
            if start_time:
                filter_expr.append(f"start_time >= {int(start_time.timestamp())}")
            if end_time:
                filter_expr.append(f"end_time <= {int(end_time.timestamp())}")
            if current_time:
                # Filter foresights where current time falls within the validity period
                current_ts = int(current_time.timestamp())
                filter_expr.append(
                    f"(start_time <= {current_ts} and end_time >= {current_ts})"
                )

            filter_str = " and ".join(filter_expr) if filter_expr else None

            # Execute search
            # Dynamically adjust ef parameter: must be >= limit, typically set to 1.5-2 times limit
            ef_value = max(128, limit * 2)  # Ensure ef >= limit, minimum 128
            # Use COSINE similarity, radius indicates returning only results with similarity >= threshold
            # Prioritize passed radius parameter, otherwise use default configuration
            similarity_radius = (
                radius if radius is not None else MILVUS_SIMILARITY_RADIUS
            )
            search_params = {"metric_type": "COSINE", "params": {"ef": ef_value}}
            # Do not set radius parameter!
            # Milvus radius is the lower bound of similarity; setting too low a value may cause issues
            # Only set when explicitly specified and > -1.0
            if radius is not None and radius > -1.0:
                search_params["params"]["radius"] = radius
            elif similarity_radius is not None and similarity_radius > -1.0:
                search_params["params"]["radius"] = similarity_radius

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
                    if hit.score >= score_threshold:
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
                            "start_time": datetime.fromtimestamp(
                                hit.entity.get("start_time", 0)
                            ),
                            "end_time": datetime.fromtimestamp(
                                hit.entity.get("end_time", 0)
                            ),
                            "duration_days": hit.entity.get("duration_days"),
                            "content": hit.entity.get("content"),
                            "evidence": hit.entity.get("evidence"),
                            "search_content": search_content,
                            "metadata": metadata,
                        }
                        search_results.append(result)

            logger.debug(
                "✅ Vector search succeeded: found %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            logger.error("❌ Vector search failed: %s", e)
            raise

    # ==================== Deletion Functionality ====================

    async def delete_by_id(self, memory_id: str) -> bool:
        """
        Delete foresight document by memory_id

        Args:
            memory_id: Unique identifier of foresight

        Returns:
            True if deletion succeeds, otherwise False
        """
        try:
            success = await super().delete_by_id(memory_id)
            if success:
                logger.debug(
                    "✅ Deleted foresight by memory_id successfully: memory_id=%s",
                    memory_id,
                )
            return success
        except Exception as e:
            logger.error(
                "❌ Failed to delete foresight by memory_id: memory_id=%s, error=%s",
                memory_id,
                e,
            )
            return False

    async def delete_by_parent_id(
        self, parent_id: str, parent_type: Optional[str] = None
    ) -> int:
        """
        Delete all associated foresights by parent memory ID and optionally parent type

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
                "✅ Deleted foresight by parent_id successfully: %s (type=%s), deleted %d records",
                parent_id,
                parent_type,
                delete_count,
            )
            return delete_count

        except Exception as e:
            logger.error("❌ Failed to delete foresight by parent_id: %s", e)
            raise

    async def delete_by_filters(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        Batch delete foresight documents based on filter conditions

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
                filter_expr.append(f"start_time >= {int(start_time.timestamp())}")
            if end_time:
                filter_expr.append(f"end_time <= {int(end_time.timestamp())}")

            if not filter_expr:
                raise ValueError("At least one filter condition must be provided")

            expr = " and ".join(filter_expr)

            # First query the number of documents to delete
            results = await self.collection.query(expr=expr, output_fields=["id"])
            delete_count = len(results)

            # Perform deletion
            await self.collection.delete(expr)

            logger.debug(
                "✅ Batch deleted foresight by filter conditions successfully: deleted %d records",
                delete_count,
            )
            return delete_count

        except Exception as e:
            logger.error(
                "❌ Failed to batch delete foresight by filter conditions: %s", e
            )
            raise
