"""
EventLogRecord Repository

提供通用事件日志的 CRUD 操作和查询功能。
"""

from datetime import datetime
from typing import List, Optional, Type, TypeVar, Union
from motor.motor_asyncio import AsyncIOMotorClientSession
from bson import ObjectId
from core.observation.logger import get_logger
from core.di.decorators import repository
from core.oxm.mongo.base_repository import BaseRepository
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
    EventLogRecordProjection,
)

# 定义泛型类型变量
T = TypeVar('T', EventLogRecord, EventLogRecordProjection)

logger = get_logger(__name__)


@repository("event_log_record_repository", primary=True)
class EventLogRecordRawRepository(BaseRepository[EventLogRecord]):
    """
    个人事件日志原始数据仓库

    提供个人事件日志的 CRUD 操作和基础查询功能。
    注意：向量应在提取时生成，此 Repository 不负责生成向量。
    """

    def __init__(self):
        super().__init__(EventLogRecord)

    # ==================== 基础 CRUD 方法 ====================

    async def save(
        self,
        event_log: EventLogRecord,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> Optional[EventLogRecord]:
        """
        保存个人事件日志

        Args:
            event_log: 个人事件日志对象
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            保存的 EventLogRecord 或 None
        """
        try:
            await event_log.insert(session=session)
            logger.info(
                "✅ 保存个人事件日志成功: id=%s, user_id=%s, parent_episode=%s",
                event_log.id,
                event_log.user_id,
                event_log.parent_episode_id,
            )
            return event_log
        except Exception as e:
            logger.error("❌ 保存个人事件日志失败: %s", e)
            return None

    async def get_by_id(
        self,
        log_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> Optional[Union[EventLogRecord, EventLogRecordProjection]]:
        """
        根据ID获取个人事件日志

        Args:
            log_id: 日志ID
            session: 可选的 MongoDB 会话，用于事务支持
            model: 返回的模型类型，默认为 EventLogRecord（完整版本），可传入 EventLogRecordShort

        Returns:
            指定类型的事件日志对象或 None
        """
        try:
            object_id = ObjectId(log_id)

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
                    "✅ 根据ID获取个人事件日志成功: %s (model=%s)",
                    log_id,
                    target_model.__name__,
                )
            else:
                logger.debug("ℹ️  未找到个人事件日志: id=%s", log_id)
            return result
        except Exception as e:
            logger.error("❌ 根据ID获取个人事件日志失败: %s", e)
            return None

    async def get_by_parent_episode_id(
        self,
        parent_episode_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[EventLogRecord, EventLogRecordProjection]]:
        """
        根据父情景记忆ID获取所有事件日志

        Args:
            parent_episode_id: 父情景记忆ID
            session: 可选的 MongoDB 会话，用于事务支持
            model: 返回的模型类型，默认为 EventLogRecord（完整版本），可传入 EventLogRecordShort

        Returns:
            指定类型的事件日志对象列表
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
                "✅ 根据父情景记忆ID获取事件日志成功: %s, 找到 %d 条记录 (model=%s)",
                parent_episode_id,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ 根据父情景记忆ID获取事件日志失败: %s", e)
            return []

    async def get_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = True,
        session: Optional[AsyncIOMotorClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[EventLogRecord, EventLogRecordProjection]]:
        """
        根据用户ID获取事件日志列表

        Args:
            user_id: 用户ID
            limit: 限制返回数量
            skip: 跳过数量
            sort_desc: 是否按时间降序排序
            session: 可选的 MongoDB 会话，用于事务支持
            model: 返回的模型类型，默认为 EventLogRecord（完整版本），可传入 EventLogRecordShort

        Returns:
            指定类型的事件日志对象列表
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

            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据用户ID获取事件日志成功: %s, 找到 %d 条记录 (model=%s)",
                user_id,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ 根据用户ID获取事件日志失败: %s", e)
            return []

    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort_desc: bool = False,
        session: Optional[AsyncIOMotorClientSession] = None,
        model: Optional[Type[T]] = None,
    ) -> List[Union[EventLogRecord, EventLogRecordProjection]]:
        """
        根据时间范围查询事件日志

        Args:
            start_time: 开始时间
            end_time: 结束时间
            user_id: 可选的用户ID过滤
            limit: 限制返回数量
            skip: 跳过数量
            sort_desc: 是否按时间降序排序，默认False（升序）
            session: 可选的 MongoDB 会话，用于事务支持
            model: 返回的模型类型，默认为 EventLogRecord（完整版本），可传入 EventLogRecordShort

        Returns:
            指定类型的事件日志对象列表
        """
        try:
            # 如果未指定 model，使用完整版本
            target_model = model if model is not None else self.model

            filter_dict = {"timestamp": {"$gte": start_time, "$lt": end_time}}
            if user_id:
                filter_dict["user_id"] = user_id

            # 根据 model 类型决定是否使用 projection
            if target_model == self.model:
                query = self.model.find(filter_dict, session=session)
            else:
                query = self.model.find(
                    filter_dict, projection_model=target_model, session=session
                )

            if sort_desc:
                query = query.sort("-timestamp")
            else:
                query = query.sort("timestamp")

            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            results = await query.to_list()
            logger.debug(
                "✅ 根据时间范围查询事件日志成功: 时间范围: %s - %s, 找到 %d 条记录 (model=%s)",
                start_time,
                end_time,
                len(results),
                target_model.__name__,
            )
            return results
        except Exception as e:
            logger.error("❌ 根据时间范围查询事件日志失败: %s", e)
            return []

    async def delete_by_id(
        self, log_id: str, session: Optional[AsyncIOMotorClientSession] = None
    ) -> bool:
        """
        根据ID删除个人事件日志

        Args:
            log_id: 日志ID
            session: 可选的 MongoDB 会话，用于事务支持

        Returns:
            是否删除成功
        """
        try:
            object_id = ObjectId(log_id)
            result = await self.model.find({"_id": object_id}, session=session).delete()
            success = result.deleted_count > 0 if result else False

            if success:
                logger.info("✅ 删除个人事件日志成功: %s", log_id)
            else:
                logger.warning("⚠️  未找到要删除的个人事件日志: %s", log_id)

            return success
        except Exception as e:
            logger.error("❌ 删除个人事件日志失败: %s", e)
            return False

    async def delete_by_parent_episode_id(
        self,
        parent_episode_id: str,
        session: Optional[AsyncIOMotorClientSession] = None,
    ) -> int:
        """
        根据父情景记忆ID删除所有事件日志

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
                "✅ 根据父情景记忆ID删除事件日志成功: %s, 删除 %d 条记录",
                parent_episode_id,
                count,
            )
            return count
        except Exception as e:
            logger.error("❌ 根据父情景记忆ID删除事件日志失败: %s", e)
            return 0


# 导出
__all__ = ["EventLogRecordRawRepository"]
