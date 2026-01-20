from typing import Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.conversation_status import (
    ConversationStatus,
)
from core.observation.logger import get_logger
from core.di.decorators import repository
from infra_layer.adapters.out.persistence.repository.dual_storage_mixin import (
    DualStorageMixin,
)

logger = get_logger(__name__)


@repository("conversation_status_raw_repository", primary=True)
class ConversationStatusRawRepository(
    DualStorageMixin,  # 添加双存储支持 - 自动拦截 MongoDB 调用
    BaseRepository[ConversationStatus],
):
    """
    Conversation status raw data repository

    Provides CRUD operations and query capabilities for conversation status data.

    Dual Storage: DualStorageMixin automatically intercepts all MongoDB operations
    """

    def __init__(self):
        super().__init__(ConversationStatus)

    # ==================== Basic CRUD Operations ====================

    async def get_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[ConversationStatus]:
        """Get conversation status by group ID"""
        try:
            result = await self.model.find_one({"group_id": group_id}, session=session)
            if result:
                logger.debug(
                    "✅ Successfully retrieved conversation status by group ID: %s",
                    group_id,
                )
            else:
                logger.debug("⚠️  Conversation status not found: group_id=%s", group_id)
            return result
        except Exception as e:
            logger.error("❌ Failed to retrieve conversation status by group ID: %s", e)
            return None

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """Delete conversation status by group ID"""
        try:
            result = await self.model.find_one({"group_id": group_id}, session=session)
            if not result:
                logger.warning(
                    "⚠️  Conversation status to delete not found: group_id=%s", group_id
                )
                return False

            await result.delete(session=session)
            logger.info(
                "✅ Successfully deleted conversation status by group ID: %s", group_id
            )
            return True
        except Exception as e:
            logger.error("❌ Failed to delete conversation status by group ID: %s", e)
            return False

    async def upsert_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[ConversationStatus]:
        """Update or insert conversation status by group ID

        Uses MongoDB atomic upsert operation to avoid concurrency race conditions.
        If a matching record is found, it updates it; otherwise, it creates a new record.
        Since group_id is unique, conversation_id will automatically use group_id as its value.

        Args:
            group_id: Group ID (will also be used as conversation_id)
            update_data: Data to update
            session: MongoDB session

        Returns:
            The updated or created conversation status record
        """
        try:
            # 1. First try to find an existing record
            existing_doc = await self.model.find_one(
                {"group_id": group_id}, session=session
            )

            if existing_doc:
                # Record found, update directly
                for key, value in update_data.items():
                    setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ Successfully updated existing conversation status: group_id=%s",
                    group_id,
                )
                print(
                    f"[ConversationStatusRawRepository] Successfully updated existing conversation status: {existing_doc}"
                )
                return existing_doc

            # 2. Record not found, try to create a new one
            try:
                new_doc = ConversationStatus(group_id=group_id, **update_data)
                await new_doc.create(session=session)
                logger.info(
                    "✅ Successfully created new conversation status: group_id=%s",
                    group_id,
                )
                print(
                    f"[ConversationStatusRawRepository] Successfully created new conversation status: {new_doc}"
                )
                return new_doc

            except Exception as create_error:
                # 3. Creation failed, check if it's a duplicate key error (concurrent case)
                error_str = str(create_error)
                if "E11000" in error_str and "duplicate key" in error_str:
                    logger.warning(
                        "⚠️  Concurrent creation conflict, re-lookup and update: group_id=%s",
                        group_id,
                    )

                    # Duplicate key error means another thread has already created the record, re-lookup and update
                    retry_doc = await self.model.find_one(
                        {"group_id": group_id}, session=session
                    )

                    if retry_doc:
                        # Found the record created by another thread, update it
                        for key, value in update_data.items():
                            setattr(retry_doc, key, value)
                        await retry_doc.save(session=session)
                        logger.debug(
                            "✅ Successfully updated after concurrency conflict: group_id=%s",
                            group_id,
                        )
                        print(
                            f"[ConversationStatusRawRepository] Successfully updated after concurrency conflict: {retry_doc}"
                        )
                        return retry_doc
                    else:
                        logger.error(
                            "❌ Still unable to find record after concurrency conflict: group_id=%s",
                            group_id,
                        )
                        return None
                else:
                    # Other types of creation errors, re-raise
                    raise create_error

        except Exception as e:
            logger.error("❌ Failed to update or create conversation status: %s", e)
            return None

    # ==================== Statistics Methods ====================

    async def count_by_group_id(
        self, group_id: str, session: Optional[AsyncClientSession] = None
    ) -> int:
        """Count the number of conversation statuses for a specified group"""
        try:
            count = await self.model.find(
                {"group_id": group_id}, session=session
            ).count()
            logger.debug(
                "✅ Successfully counted conversation statuses: group_id=%s, count=%d",
                group_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ Failed to count conversation statuses: %s", e)
            return 0
