"""
SemanticMemoryRecord Repository

提供通用语义记忆的 CRUD 操作和查询功能。
"""

from typing import List, Optional, Type, TypeVar, Union
from motor.motor_asyncio import AsyncIOMotorClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.semantic_memory_record import (
    SemanticMemoryRecord,
    SemanticMemoryRecordProjection,
)

# 定义泛型类型变量
T = TypeVar('T', SemanticMemoryRecord, SemanticMemoryRecordProjection)

logger = get_logger(__name__)


@repository("semantic_memory_record_repository", primary=True)
class SemanticMemoryRecordRawRepository(BaseRepository[SemanticMemoryRecord]):
    """
    个人语义记忆原始数据仓库

    提供个人语义记忆的 CRUD 操作和基础查询功能。
    注意：向量应在提取时生成，此 Repository 不负责生成向量。
    """

    def __init__(self):
        super().__init__(SemanticMemoryRecord)

    # ==================== 基础 CRUD 方法 ====================

    async def save(
        self,
        semantic_memory: SemanticMemoryRecord,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[SemanticMemoryRecord]:
        """
        保存个人语义记忆

        Args:
            semantic_memory: 个人语义记忆对象
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            保存的 SemanticMemoryRecord 或 None
        """
        try:
            await semantic_memory.insert(session=session)
            logger.info(
                "✅ 保存个人语义记忆成功: id=%s, user_id=%s, parent_episode=%s",
                semantic_memory.id,
                semantic_memory.user_id,
                semantic_memory.parent_episode_id,
            )
            return semantic_memory
        except Exception as e:
            logger.error("❌ 保存个人语义记忆失败: %s", e)
            return None

    async def get_by_id(
        self,
        memory_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> Optional[Union[SemanticMemoryRecord, SemanticMemoryRecordProjection]]:
        """
        根据ID获取个人语义记忆

        Args:
            memory_id: 记忆ID
            session: 可选的 MongoDB 会话，用于事务支持
            model: 返回的模型类型，默认为 SemanticMemoryRecord（完整版本），可传入 SemanticMemoryRecordShort

        Returns:
            指定类型的语义记忆对象或 None
        """
        try:
            object_id = ObjectId(memory_id)

            # 如果未指定 model，使用完整版本
            target_model = model if model is not None else self.model

            # 根据 model 类型决定是否使用 projection
            if target_model == self.model:
                result = await self.model.find_one({"_id": object_id}, session=session)
            else:
                result = await self.model.find_one(
                    {"_id": object_id}, projection_model=target_model, session=session
                )

            if result:
                logger.debug(
                    "✅ 根据ID获取个人语义记忆成功: %s (model=%s)",
                    memory_id,
                    target_model.__name__,
                )
            else:
                logger.debug("ℹ️  未找到个人语义记忆: id=%s", memory_id)
            return result
        except Exception as e:
            logger.error("❌ 根据ID获取个人语义记忆失败: %s", e)
            return None

    async def get_by_parent_episode_id(
        self,
        parent_episode_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[SemanticMemoryRecord, SemanticMemoryRecordProjection]]:
        """
        根据父情景记忆ID获取所有语义记忆

        Args:
            parent_episode_id: 父情景记忆ID
            session: 可选的 MongoDB 会话，用于事务支持
            model: 返回的模型类型，默认为 SemanticMemoryRecord（完整版本），可传入 SemanticMemoryRecordShort

        Returns:
            指定类型的语义记忆对象列表
        """
        try:
            # 如果未指定 model，使用完整版本
            target_model = model if model is not None else self.model

            # 根据 model 类型决定是否使用 projection
            if target_model == self.model:
                query = self.model.find(
                    {"parent_episode_id": parent_episode_id}, session=session
                )
            else:
                query = self.model.find(
                    {"parent_episode_id": parent_episode_id},
                    projection_model=target_model,
                    session=session,
                )

            results = await query.to_list()
            logger.debug(
                "✅ 根据父情景记忆ID获取语义记忆成功: %s, 找到 %d 条记录 (model=%s)",
                parent_episode_id,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ 根据父情景记忆ID获取语义记忆失败: %s", e)
            return []

    async def get_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        session: Optional[AsyncIOMotorClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[SemanticMemoryRecord, SemanticMemoryRecordProjection]]:
        """
        根据用户ID获取语义记忆列表

        Args:
            user_id: 用户ID
            limit: 限制返回数量
            skip: 跳过数量
            session: 可选的 MongoDB 会话，用于事务支持
            model: 返回的模型类型，默认为 SemanticMemoryRecord（完整版本），可传入 SemanticMemoryRecordShort

        Returns:
            指定类型的语义记忆对象列表
        """
        try:
            # 如果未指定 model，使用完整版本
            target_model = model if model is not None else self.model

            # 根据 model 类型决定是否使用 projection
            if target_model == self.model:
                query = self.model.find({"user_id": user_id}, session=session)
            else:
                query = self.model.find(
                    {"user_id": user_id}, projection_model=target_model, session=session
                )

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据用户ID获取语义记忆成功: %s, 找到 %d 条记录 (model=%s)",
                user_id,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户ID获取语义记忆失败: %s", e)
            return []

    async def delete_by_id(
        self, memory_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        根据ID删除个人语义记忆

        Args:
            memory_id: 记忆ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否删除成功
        """
        try:
            object_id = ObjectId(memory_id)
            result = await self.model.find({"_id": object_id}, session=session).delete()
            success = result.deleted_count > 0 if result else False

            if success:
                logger.info("✅ 删除个人语义记忆成功: %s", memory_id)
            else:
                logger.warning("⚠️  未找到要删除的个人语义记忆: %s", memory_id)

            return success
        except Exception as e:
            logger.error("❌ 删除个人语义记忆失败: %s", e)
            return False

    async def delete_by_parent_episode_id(
        self,
        parent_episode_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> int:
        """
        根据父情景记忆ID删除所有语义记忆

        Args:
            parent_episode_id: 父情景记忆ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            删除的记录数量
        """
        try:
            result = await self.model.find(
                {"parent_episode_id": parent_episode_id}, session=session
            ).delete()
            count = result.deleted_count if result else 0
            logger.info(
                "✅ 根据父情景记忆ID删除语义记忆成功: %s, 删除 %d 条记录",
                parent_episode_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ 根据父情景记忆ID删除语义记忆失败: %s", e)
            return 0


# 导出
__all__ = ["SemanticMemoryRecordRawRepository"]
