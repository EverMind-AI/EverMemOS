"""
ConversationMeta Raw Repository

提供对话元数据的数据库操作接口
"""

import logging
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClientSession

from core.oxm.mongo.base_repository import BaseRepository
from core.di.decorators import repository
from infra_layer.adapters.out.persistence.document.memory.conversation_meta import (
    ConversationMeta,
)

logger = logging.getLogger(__name__)


@repository("conversation_meta_raw_repository", primary=True)
class ConversationMetaRawRepository(BaseRepository[ConversationMeta]):
    """
    对话元数据原始仓储层

    提供对话元数据的基础数据库操作
    """

    def __init__(self):
        """初始化仓储"""
        super().__init__(ConversationMeta)

    async def get_by_group_id(
        self, group_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> Optional[ConversationMeta]:
        """
        根据群组ID获取对话元数据

        Args:
            group_id: 群组ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            对话元数据对象或 None
        """
        try:
            conversation_meta = await self.model.find_one(
                {"group_id": group_id}, session=session
            )
            if conversation_meta:
                logger.debug("✅ 根据 group_id 获取对话元数据成功: %s", group_id)
            return conversation_meta
        except Exception as e:
            logger.error("❌ 根据 group_id 获取对话元数据失败: %s", e)
            return None

    async def list_by_scene(
        self,
        scene: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> List[ConversationMeta]:
        """
        根据场景标识获取对话元数据列表

        Args:
            scene: 场景标识
            limit: 返回数量限制
            skip: 跳过数量
            session: 可选的 MongoDB 会话

        Returns:
            对话元数据列表
        """
        try:
            query = self.model.find({"scene": scene}, session=session)
            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            result = await query.to_list()
            logger.debug("✅ 根据场景获取对话元数据列表成功: scene=%s, count=%d", scene, len(result))
            return result
        except Exception as e:
            logger.error("❌ 根据场景获取对话元数据列表失败: %s", e)
            return []

    async def create_conversation_meta(
        self,
        conversation_meta: ConversationMeta,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[ConversationMeta]:
        """
        创建新的对话元数据

        Args:
            conversation_meta: 对话元数据对象
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            创建的对话元数据对象或 None
        """
        try:
            await conversation_meta.insert(session=session)
            logger.info(
                "✅ 创建对话元数据成功: group_id=%s, scene=%s",
                conversation_meta.group_id,
                conversation_meta.scene,
            )
            return conversation_meta
        except Exception as e:
            logger.error("❌ 创建对话元数据失败: %s", e, exc_info=True)
            return None

    async def update_by_group_id(
        self,
        group_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[ConversationMeta]:
        """
        根据群组ID更新对话元数据

        Args:
            group_id: 群组ID
            update_data: 更新数据字典
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            更新后的对话元数据对象或 None
        """
        try:
            conversation_meta = await self.get_by_group_id(group_id, session=session)
            if conversation_meta:
                for key, value in update_data.items():
                    if hasattr(conversation_meta, key):
                        setattr(conversation_meta, key, value)
                await conversation_meta.save(session=session)
                logger.debug("✅ 根据 group_id 更新对话元数据成功: %s", group_id)
                return conversation_meta
            return None
        except Exception as e:
            logger.error("❌ 根据 group_id 更新对话元数据失败: %s", e, exc_info=True)
            return None

    async def upsert_by_group_id(
        self,
        group_id: str,
        conversation_data: Dict[str, Any],
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[ConversationMeta]:
        """
        根据群组ID更新或插入对话元数据

        使用 MongoDB 原子 upsert 操作来避免并发竞态条件

        Args:
            group_id: 群组ID
            conversation_data: 对话元数据字典
            session: 可选的 MongoDB 会话

        Returns:
            更新或创建的对话元数据对象
        """
        try:
            # 1. 首先尝试查找现有记录
            existing_doc = await self.model.find_one(
                {"group_id": group_id}, session=session
            )

            if existing_doc:
                # 找到记录，直接更新
                for key, value in conversation_data.items():
                    if hasattr(existing_doc, key):
                        setattr(existing_doc, key, value)
                await existing_doc.save(session=session)
                logger.debug("✅ 更新现有对话元数据成功: group_id=%s", group_id)
                return existing_doc

            # 2. 没找到记录，创建新记录
            try:
                new_doc = ConversationMeta(group_id=group_id, **conversation_data)
                await new_doc.insert(session=session)
                logger.info("✅ 创建新对话元数据成功: group_id=%s", group_id)
                return new_doc
            except Exception as create_error:
                logger.error("❌ 创建对话元数据失败: %s", create_error, exc_info=True)
                return None

        except Exception as e:
            logger.error("❌ Upsert 对话元数据失败: %s", e, exc_info=True)
            return None

    async def delete_by_group_id(
        self, group_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        根据群组ID删除对话元数据

        Args:
            group_id: 群组ID
            session: 可选的 MongoDB 会话

        Returns:
            是否删除成功
        """
        try:
            result = await self.model.find_one(
                {"group_id": group_id}, session=session
            ).delete()
            if result:
                logger.info("✅ 删除对话元数据成功: group_id=%s", group_id)
                return True
            return False
        except Exception as e:
            logger.error("❌ 删除对话元数据失败: %s", e)
            return False

