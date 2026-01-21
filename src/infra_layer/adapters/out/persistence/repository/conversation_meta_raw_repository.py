"""
ConversationMeta Raw Repository

Provides database operation interfaces for conversation metadata
"""

import logging
from typing import Optional, List, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession

from core.oxm.mongo.base_repository import BaseRepository
from core.di.decorators import repository
from core.constants.exceptions import ValidationException
from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
    ConversationMeta,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_mixin import (
    DualStorageMixin,
)
from memory_layer.profile_manager.config import ScenarioType

logger = logging.getLogger(__name__)

# Allowed scene enum values (derived from ScenarioType)
ALLOWED_SCENES = [e.value for e in ScenarioType]


@repository("conversation_meta_raw_repository", primary=True)
class ConversationMetaRawRepository(
    DualStorageMixin,  # 添加双存储支持 - 自动拦截 MongoDB 调用
    BaseRepository[ConversationMeta],
):
    """
    Raw repository layer for conversation metadata

    Provides basic database operations for conversation metadata

    Dual Storage: DualStorageMixin automatically intercepts all MongoDB operations
    """

    def __init__(self):
        """Initialize repository"""
        super().__init__(ConversationMeta)

    def _validate_scene(self, scene: str) -> None:
        """
        Validate if scene is valid

        Args:
            scene: Scene identifier

        Raises:
            ValidationException: When scene validation fails
        """
        if scene not in ALLOWED_SCENES:
            error_message = (
                f"invalid scene value: {scene}, "
                f"allowed values: {ALLOWED_SCENES}"
            )
            logger.error("❌ Scene validation failed: %s", error_message)
            raise ValidationException(
                message=error_message,
                field="scene",
                details={
                    "invalid_value": scene,
                    "allowed_values": ALLOWED_SCENES,
                },
            )

    async def get_by_group_id(
        self, group_id: Optional[str], session: Optional[AsyncClientSession] = None
    ) -> Optional[ConversationMeta]:
        """
        Get conversation metadata by group ID with automatic fallback to default config

        Args:
            group_id: Group ID (can be None to get default config directly)
            session: Optional MongoDB session, used for transaction support

        Returns:
            Conversation metadata object or None.
            If group_id is provided but not found, automatically falls back to default config.
        """
        try:
            # First try to find by exact group_id
            conversation_meta = await self.model.find_one(
                {"group_id": group_id}, session=session
            )
            if conversation_meta:
                logger.debug(
                    "✅ Successfully retrieved conversation metadata by group_id: %s",
                    group_id,
                )
                return conversation_meta

            # If group_id is None or not found, no fallback needed for None case
            if group_id is None:
                logger.debug("⚠️ Default conversation metadata not found")
                return None

            # Fallback to default config (group_id is None)
            logger.debug(
                "⚡ group_id %s not found, falling back to default config", group_id
            )
            default_meta = await self.model.find_one(
                {"group_id": None}, session=session
            )
            if default_meta:
                logger.debug("✅ Using default conversation metadata")
            else:
                logger.debug("⚠️ No default conversation metadata found")
            return default_meta

        except Exception as e:
            logger.error(
                "❌ Failed to retrieve conversation metadata by group_id: %s", e
            )
            return None

    async def list_by_scene(
        self,
        scene: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        session: Optional[AsyncClientSession] = None,
    ) -> List[ConversationMeta]:
        """
        Get list of conversation metadata by scene identifier

        Args:
            scene: Scene identifier
            limit: Limit on number of returned items
            skip: Number of items to skip
            session: Optional MongoDB session

        Returns:
            List of conversation metadata
        """
        try:
            # Validate scene field
            self._validate_scene(scene=scene)

            query = self.model.find({"scene": scene}, session=session)
            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            result = await query.to_list()
            logger.debug(
                "✅ Successfully retrieved conversation metadata list by scene: scene=%s, count=%d",
                scene,
                len(result),
            )
            return result
        except ValidationException:
            # Re-raise ValidationException to propagate detailed error info
            raise
        except Exception as e:
            logger.error(
                "❌ Failed to retrieve conversation metadata list by scene: %s", e
            )
            return []

    async def create_conversation_meta(
        self,
        conversation_meta: ConversationMeta,
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[ConversationMeta]:
        """
        Create new conversation metadata

        Args:
            conversation_meta: Conversation metadata object
            session: Optional MongoDB session, used for transaction support

        Returns:
            Created conversation metadata object or None
        """
        try:
            # Validate scene field
            self._validate_scene(scene=conversation_meta.scene)

            await conversation_meta.insert(session=session)
            logger.info(
                "✅ Successfully created conversation metadata: group_id=%s, scene=%s",
                conversation_meta.group_id,
                conversation_meta.scene,
            )
            return conversation_meta
        except ValidationException:
            # Re-raise ValidationException to propagate detailed error info
            raise
        except Exception as e:
            logger.error(
                "❌ Failed to create conversation metadata: %s", e, exc_info=True
            )
            return None

    async def update_by_group_id(
        self,
        group_id: Optional[str],
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[ConversationMeta]:
        """
        Update conversation metadata by group ID

        Args:
            group_id: Group ID (can be None for default config)
            update_data: Dictionary of update data
            session: Optional MongoDB session, used for transaction support

        Returns:
            Updated conversation metadata object or None

        Raises:
            ValidationException: When scene validation fails
        """
        try:
            # Validate scene if present in update data
            if "scene" in update_data:
                self._validate_scene(update_data["scene"])

            conversation_meta = await self.get_by_group_id(group_id, session=session)
            if conversation_meta:
                for key, value in update_data.items():
                    if hasattr(conversation_meta, key):
                        setattr(conversation_meta, key, value)
                await conversation_meta.save(session=session)
                logger.debug(
                    "✅ Successfully updated conversation metadata by group_id: %s",
                    group_id,
                )
                return conversation_meta
            return None
        except ValidationException:
            # Re-raise ValidationException to propagate detailed error info
            raise
        except Exception as e:
            logger.error(
                "❌ Failed to update conversation metadata by group_id: %s",
                e,
                exc_info=True,
            )
            return None

    async def upsert_by_group_id(
        self,
        group_id: Optional[str],
        conversation_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[ConversationMeta]:
        """
        Update or insert conversation metadata by group ID

        Uses MongoDB atomic upsert operation to avoid concurrency race conditions

        Args:
            group_id: Group ID (can be None for default config)
            conversation_data: Conversation metadata dictionary
            session: Optional MongoDB session

        Returns:
            Updated or created conversation metadata object

        Raises:
            ValidationException: When scene validation fails
        """
        try:
            # Validate scene if present in conversation data
            if "scene" in conversation_data:
                self._validate_scene(conversation_data["scene"])

            # 1. First try to find existing record
            existing_doc = await self.model.find_one(
                {"group_id": group_id}, session=session
            )

            if existing_doc:
                # Found record, update directly
                for key, value in conversation_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug(
                    "✅ Successfully updated existing conversation metadata: group_id=%s",
                    group_id,
                )
                return existing_doc

            # 2. No record found, create new one
            try:
                new_doc = ConversationMeta(group_id=group_id, **conversation_data)
                await new_doc.insert(session=session)
                logger.info(
                    "✅ Successfully created new conversation metadata: group_id=%s (is_default=%s)",
                    group_id,
                    group_id is None,
                )
                return new_doc
            except Exception as create_error:
                logger.error(
                    "❌ Failed to create conversation metadata: %s",
                    create_error,
                    exc_info=True,
                )
                return None

        except ValidationException:
            # Re-raise ValidationException to propagate detailed error info
            raise
        except Exception as e:
            logger.error(
                "❌ Failed to upsert conversation metadata: %s", e, exc_info=True
            )
            return None

    async def delete_by_group_id(
        self, group_id: Optional[str], session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete conversation metadata by group ID

        Args:
            group_id: Group ID (can be None for default config)
            session: Optional MongoDB session

        Returns:
            Whether deletion was successful
        """
        try:
            result = await self.model.find(
                {"group_id": group_id}, session=session
            ).delete()
            if result and result.deleted_count > 0:
                logger.info(
                    "✅ Successfully deleted conversation metadata: group_id=%s",
                    group_id,
                )
                return True
            return False
        except Exception as e:
            logger.error("❌ Failed to delete conversation metadata: %s", e)
            return False
