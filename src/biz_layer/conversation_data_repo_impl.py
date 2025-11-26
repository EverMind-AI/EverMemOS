"""
ConversationDataRepository 真实数据库实现

基于Redis缓存的对话数据访问实现，使用Redis长度限制缓存管理器存储RawData对象
"""

from typing import List, Optional
from core.observation.logger import get_logger
from core.di.decorators import repository
from biz_layer.conversation_data_repo import ConversationDataRepository
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from biz_layer.mem_db_operations import _normalize_datetime_for_storage
from common_utils.datetime_utils import get_now_with_timezone
from core.di import get_bean

logger = get_logger(__name__)


@repository("conversation_data_repo", primary=True)
class ConversationDataRepositoryImpl(ConversationDataRepository):
    """
    ConversationDataRepository 真实数据库实现

    基于Redis长度限制缓存管理器的对话数据访问实现，将RawData对象直接存储到Redis中
    """

    def __init__(self):
        """初始化Repository"""
        # 获取Redis长度限制缓存管理器工厂
        self._cache_factory = get_bean("redis_length_cache_factory")
        self._cache_manager = None

    async def _get_cache_manager(self):
        """获取缓存管理器实例"""
        if self._cache_manager is None:
            # 创建缓存管理器：最大长度1000，过期时间60分钟，清理概率0.1
            self._cache_manager = await self._cache_factory.create_cache_manager(
                max_length=1000, expire_minutes=60, cleanup_probability=0.1
            )
        return self._cache_manager

    def _get_redis_key(self, group_id: str) -> str:
        """生成Redis键名"""
        return f"conversation_data:{group_id}"

    async def save_conversation_data(
        self, raw_data_list: List[RawData], group_id: str
    ) -> bool:
        """
        保存对话数据到Redis缓存

        将RawData对象直接序列化并存储到Redis长度限制队列中

        Args:
            raw_data_list: RawData列表
            group_id: 群组ID

        Returns:
            bool: 保存成功返回True，失败返回False
        """
        if not raw_data_list:
            logger.info("没有对话数据需要保存: group_id=%s", group_id)
            return True

        logger.info(
            "开始保存对话数据到Redis: group_id=%s, 数量=%d",
            group_id,
            len(raw_data_list),
        )

        try:
            cache_manager = await self._get_cache_manager()
            redis_key = self._get_redis_key(group_id)
            saved_count = 0

            for raw_data in raw_data_list:
                try:
                    # 从RawData中提取时间戳
                    timestamp = None
                    if raw_data.content:
                        timestamp = raw_data.content.get(
                            'timestamp'
                        ) or raw_data.content.get('createTime')

                    # 确保时间戳是datetime对象
                    if timestamp:
                        timestamp = _normalize_datetime_for_storage(timestamp)
                    else:
                        timestamp = get_now_with_timezone()

                    # 使用Redis长度限制缓存管理器追加数据
                    # 直接将RawData对象传入，缓存管理器会处理序列化
                    success = await cache_manager.append(
                        redis_key,
                        raw_data.to_json(),  # 存储序列化的JSON字符串
                        timestamp=timestamp,
                    )

                    if success:
                        saved_count += 1
                        logger.debug(
                            "保存RawData到Redis成功: data_id=%s", raw_data.data_id
                        )
                    else:
                        logger.error(
                            "保存RawData到Redis失败: data_id=%s", raw_data.data_id
                        )

                except (ValueError, TypeError, AttributeError) as e:
                    logger.error("处理单条RawData失败: %s", e)
                    # 继续处理下一条数据，不中断整个流程
                    continue

            logger.info(
                "对话数据保存到Redis完成: group_id=%s, 成功保存=%d/%d",
                group_id,
                saved_count,
                len(raw_data_list),
            )
            return saved_count > 0

        except (
            RuntimeError,
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
        ) as e:
            logger.error("保存对话数据到Redis失败: group_id=%s, 错误=%s", group_id, e)
            return False

    async def get_conversation_data(
        self,
        group_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[RawData]:
        """
        从Redis缓存获取对话数据

        从Redis长度限制队列中读取JSON数据并反序列化为RawData对象

        Args:
            group_id: 群组ID
            start_time: 开始时间（ISO格式字符串）
            end_time: 结束时间（ISO格式字符串）
            limit: 限制返回数量

        Returns:
            List[RawData]: 对话数据列表
        """
        logger.info(
            "开始从Redis获取对话数据: group_id=%s, start_time=%s, end_time=%s, limit=%d",
            group_id,
            start_time,
            end_time,
            limit,
        )

        raw_data_list: List[RawData] = []

        try:
            cache_manager = await self._get_cache_manager()
            redis_key = self._get_redis_key(group_id)

            # 转换时间格式为datetime对象
            start_dt = (
                _normalize_datetime_for_storage(start_time) if start_time else None
            )
            end_dt = _normalize_datetime_for_storage(end_time) if end_time else None

            # 使用缓存管理器的按时间戳范围获取方法
            try:
                # 使用缓存管理器的新方法获取指定时间范围内的数据
                cache_data_list = await cache_manager.get_by_timestamp_range(
                    redis_key,
                    start_timestamp=start_dt,
                    end_timestamp=end_dt,
                    limit=limit,
                )

                logger.debug("从Redis按时间范围获取到 %d 条消息", len(cache_data_list))

                # 按时间戳升序排序，越早的越靠前
                cache_data_list = sorted(cache_data_list, key=lambda x: x.get("timestamp", 0))
                # 反序列化消息为RawData对象
                for cache_data in cache_data_list:
                    try:
                        # 从缓存数据中提取JSON字符串
                        json_data = cache_data.get("data")
                        if json_data is None:
                            logger.warning("缓存数据中缺少data字段: %s", cache_data)
                            continue

                        # 如果数据已经是字符串，直接使用；否则转换为字符串
                        if isinstance(json_data, dict):
                            # 如果是字典，说明缓存管理器已经解析了JSON，需要重新序列化
                            import json

                            json_str = json.dumps(json_data, ensure_ascii=False)
                        else:
                            json_str = str(json_data)

                        # 反序列化RawData对象
                        raw_data = RawData.from_json_str(json_str)
                        raw_data_list.append(raw_data)

                    except (ValueError, TypeError, AttributeError) as e:
                        logger.error("反序列化RawData失败: %s", e)
                        continue

                logger.debug("成功反序列化 %d 条RawData对象", len(raw_data_list))

            except (
                RuntimeError,
                ConnectionError,
                TimeoutError,
                ValueError,
                TypeError,
            ) as e:
                logger.error("从Redis获取数据时出错: %s", e)
                return []

            logger.info(
                "从Redis获取对话数据完成: group_id=%s, 返回%d条数据",
                group_id,
                len(raw_data_list),
            )
            return raw_data_list

        except (
            RuntimeError,
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
        ) as e:
            logger.error("从Redis获取对话数据失败: group_id=%s, 错误=%s", group_id, e)
            return []

    async def delete_conversation_data(self, group_id: str) -> bool:
        """
        删除指定群组的所有对话数据

        清空Redis中该群组的所有缓存消息，通常用于边界判断后重置对话历史

        Args:
            group_id: 群组ID

        Returns:
            bool: 删除成功返回True，失败返回False
        """
        logger.info("开始删除对话数据: group_id=%s", group_id)

        try:
            cache_manager = await self._get_cache_manager()
            redis_key = self._get_redis_key(group_id)

            # 使用缓存管理器删除整个键
            success = await cache_manager.clear_queue(redis_key)

            if success:
                logger.info("成功删除对话数据: group_id=%s", group_id)
            else:
                logger.warning("删除对话数据失败或键不存在: group_id=%s", group_id)

            return success

        except (
            RuntimeError,
            ConnectionError,
            TimeoutError,
            ValueError,
            TypeError,
        ) as e:
            logger.error("删除对话数据失败: group_id=%s, 错误=%s", group_id, e)
            return False
