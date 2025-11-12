"""
Kafka Consumer 工厂

基于 kafka_topic、group_id、server 提供 AIOKafkaConsumer 缓存和管理功能。
支持 force_new 逻辑创建全新的 consumer 实例。
"""

import asyncio
import json
import ssl
import os
from typing import Dict, List, Optional, Any
from hashlib import md5

import bson
from aiokafka import AIOKafkaConsumer

from component.config_provider import ConfigProvider
from core.di.decorators import component
from core.observation.logger import get_logger
from common_utils.project_path import CURRENT_DIR
from common_utils.datetime_utils import from_iso_format, to_timestamp
from core.di.utils import get_bean_by_type

logger = get_logger(__name__)


def get_ca_file_path(ca_file_path: str) -> Optional[str]:
    """
    获取 CA 证书文件的完整路径

    基于 CURRENT_DIR + /config/kafka/ca/ 构造路径

    Returns:
        Optional[str]: CA 证书文件路径，如果不存在则返回 None
    """
    # CURRENT_DIR 指向 src 目录，向上一级到项目根目录，再进入 config 目录
    ca_full_path = CURRENT_DIR / "config" / ca_file_path

    if ca_full_path.exists():
        logger.info("使用默认 CA 证书文件: %s", ca_full_path)
        return str(ca_full_path)
    else:
        logger.warning("默认 CA 证书文件不存在: %s", ca_full_path)
        return None


def get_default_kafka_config() -> Dict[str, Any]:
    """
    基于环境变量获取默认的 Kafka Consumer 配置

    环境变量：
    - KAFKA_SERVERS: Kafka 服务器列表，逗号分隔
    - KAFKA_TOPIC: Kafka 主题
    - KAFKA_GROUP_ID: Consumer 组 ID
    - MAX_POLL_INTERVAL_MS: 最大轮询间隔（毫秒）
    - SESSION_TIMEOUT_MS: 会话超时时间（毫秒）
    - HEARTBEAT_INTERVAL_MS: 心跳间隔（毫秒）
    - CA_FILE_PATH: CA证书文件路径

    Returns:
        Dict[str, Any]: 配置字典
    """
    # 获取环境变量，提供默认值
    kafka_servers_str = os.getenv("KAFKA_SERVERS", "")
    kafka_servers = [server.strip() for server in kafka_servers_str.split(",")]

    kafka_topic = os.getenv("KAFKA_TOPIC", "")
    kafka_group_id = os.getenv("KAFKA_GROUP_ID", "aic_test_0908")
    max_poll_interval_ms = int(os.getenv("MAX_POLL_INTERVAL_MS", "3600000"))
    session_timeout_ms = int(os.getenv("SESSION_TIMEOUT_MS", "10000"))
    heartbeat_interval_ms = int(os.getenv("HEARTBEAT_INTERVAL_MS", "3000"))

    # 处理CA证书路径
    ca_file_path = None
    if os.getenv("CA_FILE_PATH"):
        ca_file_path = get_ca_file_path(
            os.getenv("CA_FILE_PATH", "kafka/ca_file/AmazonRootCA1.pem")
        )

    config = {
        "kafka_servers": kafka_servers,
        "kafka_topic": kafka_topic,
        "kafka_group_id": kafka_group_id,
        "max_poll_interval_ms": max_poll_interval_ms,
        "session_timeout_ms": session_timeout_ms,
        "heartbeat_interval_ms": heartbeat_interval_ms,
        "ca_file_path": ca_file_path,
        "auto_offset_reset": "earliest",
        "enable_auto_commit": True,
    }

    logger.info("获取默认 Kafka Consumer 配置:")
    logger.info("  服务器: %s", kafka_servers)
    logger.info("  主题: %s", kafka_topic)
    logger.info("  组ID: %s", kafka_group_id)
    logger.info("  最大轮询间隔: %s ms", max_poll_interval_ms)
    logger.info("  会话超时: %s ms", session_timeout_ms)
    logger.info("  心跳间隔: %s ms", heartbeat_interval_ms)
    logger.info("  CA证书: %s", ca_file_path or "无")

    return config


def get_cache_key(
    kafka_servers: List[str], kafka_topic: str, kafka_group_id: str
) -> str:
    """
    生成缓存键
    基于 servers、topic、group_id 生成唯一标识

    Args:
        kafka_servers: Kafka服务器列表
        kafka_topic: Kafka主题
        kafka_group_id: Consumer组ID

    Returns:
        str: 缓存键
    """
    servers_str = ",".join(sorted(kafka_servers))
    key_content = f"{servers_str}:{kafka_topic}:{kafka_group_id}"
    return md5(key_content.encode()).hexdigest()


def get_consumer_name(kafka_topic: str, kafka_group_id: str) -> str:
    """
    获取 consumer 名称

    Args:
        kafka_topic: Kafka主题
        kafka_group_id: Consumer组ID

    Returns:
        str: Consumer名称
    """
    # 使用短横线连接多个topic名称，避免名称过长
    topic_str = "-".join(topic.strip() for topic in kafka_topic.split(","))
    return f"{topic_str}.{kafka_group_id}"


def json_bson_decode(value: bytes | None) -> Any:
    """
    JSON/BSON 解码器
    优先尝试 JSON 解码，失败时尝试 BSON 解码
    """
    if not value or value == b"null":
        return value
    try:
        return json.loads(value.decode("utf-8"))
    except Exception:
        try:
            return bson.decode(value)
        except Exception as e:
            logger.error("bson解析错误: %s", e)
            return value


@component(name="kafka_consumer_factory", primary=True)
class KafkaConsumerFactory:
    """
    Kafka Consumer 工厂
    ### AIOKafkaConsumer是有状态的，因此不能在多个地方使用同一个实例 ###

    提供基于配置的 AIOKafkaConsumer 缓存和管理功能
    支持 force_new 参数创建全新的 consumer 实例
    """

    def __init__(self):
        """初始化 Kafka Consumer 工厂"""
        self._consumers: Dict[str, AIOKafkaConsumer] = {}
        self._lock = asyncio.Lock()
        self._default_config: Optional[Dict[str, Any]] = None
        logger.info("KafkaConsumerFactory initialized")

    async def create_consumer(
        self,
        kafka_servers: List[str],
        kafka_topic: str,
        kafka_group_id: str,
        ca_file_path: Optional[str] = None,
        max_poll_interval_ms: int = 300000,
        session_timeout_ms: int = 10000,
        heartbeat_interval_ms: int = 3000,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = True,
    ) -> AIOKafkaConsumer:
        """
        创建 AIOKafkaConsumer 实例

        Args:
            kafka_servers: Kafka服务器列表
            kafka_topic: Kafka主题
            kafka_group_id: Consumer组ID
            ca_file_path: CA证书文件路径
            max_poll_interval_ms: 最大轮询间隔（毫秒）
            session_timeout_ms: 会话超时时间（毫秒）
            heartbeat_interval_ms: 心跳间隔（毫秒）
            auto_offset_reset: 自动偏移重置策略
            enable_auto_commit: 是否启用自动提交

        Returns:
            AIOKafkaConsumer 实例
        """
        # 创建 SSL 上下文
        ssl_context = None
        if ca_file_path:
            config_provider = get_bean_by_type(ConfigProvider)
            ca_file_content = config_provider.get_raw_config(ca_file_path)
            ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
            ssl_context.load_verify_locations(cadata=ca_file_content)

        # 处理可能的多个topic（逗号分隔）
        topics = [topic.strip() for topic in kafka_topic.split(",")]
        consumer_name = get_consumer_name(kafka_topic, kafka_group_id)

        # 创建 AIOKafkaConsumer
        consumer = AIOKafkaConsumer(
            *topics,  # 解包topics列表作为独立参数
            bootstrap_servers=kafka_servers,
            group_id=kafka_group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            value_deserializer=json_bson_decode,
            security_protocol="SSL" if ca_file_path else "PLAINTEXT",
            ssl_context=ssl_context,
            max_poll_interval_ms=max_poll_interval_ms,
            session_timeout_ms=session_timeout_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
        )

        logger.info("Created AIOKafkaConsumer for %s", consumer_name)
        return consumer

    async def get_consumer(
        self,
        kafka_servers: List[str],
        kafka_topic: str,
        kafka_group_id: str,
        force_new: bool = False,
        **kwargs,
    ) -> AIOKafkaConsumer:
        """
        获取 AIOKafkaConsumer 实例

        Args:
            kafka_servers: Kafka服务器列表
            kafka_topic: Kafka主题
            kafka_group_id: Consumer组ID
            force_new: 是否强制创建新实例，默认 False
            **kwargs: 其他配置参数

        Returns:
            AIOKafkaConsumer 实例
        """
        cache_key = get_cache_key(kafka_servers, kafka_topic, kafka_group_id)
        consumer_name = get_consumer_name(kafka_topic, kafka_group_id)

        async with self._lock:
            # 如果强制创建新实例，或者缓存中不存在
            if force_new or cache_key not in self._consumers:
                logger.info(
                    "Creating new consumer for %s (force_new=%s)",
                    consumer_name,
                    force_new,
                )

                # 如果是强制创建新实例，先清理旧实例
                if force_new and cache_key in self._consumers:
                    old_consumer = self._consumers[cache_key]
                    try:
                        await old_consumer.stop()
                    except Exception as e:
                        logger.error("Error stopping old consumer: %s", e)

                # 创建新的 consumer 实例
                consumer = await self.create_consumer(
                    kafka_servers=kafka_servers,
                    kafka_topic=kafka_topic,
                    kafka_group_id=kafka_group_id,
                    **kwargs,
                )
                self._consumers[cache_key] = consumer

                logger.info(
                    "Consumer %s created and cached with key %s",
                    consumer_name,
                    cache_key,
                )
            else:
                consumer = self._consumers[cache_key]
                logger.debug("Using cached consumer for %s", consumer_name)

        return consumer

    async def get_default_consumer(self, force_new: bool = False) -> AIOKafkaConsumer:
        """
        获取基于环境变量配置的默认 AIOKafkaConsumer 实例

        Args:
            force_new: 是否强制创建新实例，默认 False

        Returns:
            AIOKafkaConsumer 实例
        """
        # 获取或创建默认配置
        if self._default_config is None:
            self._default_config = get_default_kafka_config()

        config = self._default_config
        return await self.get_consumer(
            kafka_servers=config["kafka_servers"],
            kafka_topic=config["kafka_topic"],
            kafka_group_id=config["kafka_group_id"],
            force_new=force_new,
            ca_file_path=config.get("ca_file_path"),
            max_poll_interval_ms=config.get("max_poll_interval_ms", 20 * 60 * 1000),
            session_timeout_ms=config.get("session_timeout_ms", 10000),
            heartbeat_interval_ms=config.get("heartbeat_interval_ms", 3000),
            auto_offset_reset=config.get("auto_offset_reset", "earliest"),
            enable_auto_commit=True,
        )

    async def remove_consumer(
        self, kafka_servers: List[str], kafka_topic: str, kafka_group_id: str
    ) -> bool:
        """
        移除指定的 consumer

        Args:
            kafka_servers: Kafka服务器列表
            kafka_topic: Kafka主题
            kafka_group_id: Consumer组ID

        Returns:
            bool: 是否成功移除
        """
        cache_key = get_cache_key(kafka_servers, kafka_topic, kafka_group_id)
        consumer_name = get_consumer_name(kafka_topic, kafka_group_id)

        async with self._lock:
            if cache_key in self._consumers:
                consumer = self._consumers[cache_key]
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.error("Error stopping consumer during removal: %s", e)

                del self._consumers[cache_key]
                logger.info("Consumer %s removed from cache", consumer_name)
                return True
            else:
                logger.warning("Consumer %s not found in cache", consumer_name)
                return False

    async def clear_all_consumers(self) -> None:
        """清理所有缓存的 consumer"""
        async with self._lock:
            for cache_key, consumer in self._consumers.items():
                try:
                    await consumer.stop()
                except Exception as e:
                    logger.error("Error stopping consumer %s: %s", cache_key, e)

            self._consumers.clear()
            logger.info("All consumers cleared from cache")

    async def seek_to_datetime(
        self, offset_datetime: str, consumer: AIOKafkaConsumer
    ) -> bool:
        """
        基于时间格式调整 Kafka Consumer 的 offset

        Args:
            offset_datetime: 时间字符串，格式为 "2025-09-23 15:21:12"
            consumer: AIOKafkaConsumer 实例

        Returns:
            bool: 是否成功调整 offset

        Raises:
            ValueError: 时间格式不正确
            RuntimeError: Consumer 未启动或调整 offset 失败
        """
        try:
            # 解析时间字符串为带时区的 datetime 对象
            target_dt = from_iso_format(offset_datetime)
            # 转换为毫秒级时间戳（Kafka 使用毫秒级时间戳）
            target_timestamp_ms = int(to_timestamp(target_dt) * 1000)

            logger.info(
                "Seeking consumer to datetime: %s (timestamp: %d)",
                offset_datetime,
                target_timestamp_ms,
            )

            # 检查 consumer 是否已启动并获取分区分配
            try:
                # 尝试获取分区分配，如果 consumer 未启动会抛出异常
                partitions = consumer.assignment()
                if not partitions:
                    raise RuntimeError(
                        "Consumer has no assigned partitions. Make sure consumer is started and has subscribed to topics."
                    )
            except Exception as e:
                raise RuntimeError(
                    "Consumer must be started before seeking to timestamp"
                ) from e

            # 为每个分区构建时间戳映射
            timestamp_map = {partition: target_timestamp_ms for partition in partitions}

            # 使用 offsets_for_times 获取对应时间戳的 offset
            offset_map = await consumer.offsets_for_times(timestamp_map)

            # 统计各topic的处理情况
            topic_stats = {}
            seek_count = 0

            # 处理每个分区
            for partition in partitions:
                topic_name = partition.topic
                if topic_name not in topic_stats:
                    topic_stats[topic_name] = {
                        'total_partitions': 0,
                        'found_offsets': 0,
                        'used_latest': 0,
                    }
                topic_stats[topic_name]['total_partitions'] += 1

                # 获取该分区的offset信息
                offset_info = offset_map.get(partition) if offset_map else None

                if offset_info is not None:
                    # 找到了对应时间戳的offset
                    target_offset = offset_info.offset
                    consumer.seek(partition, target_offset)
                    seek_count += 1
                    topic_stats[topic_name]['found_offsets'] += 1
                    logger.info(
                        "Seeked partition %s (topic: %s) to offset %d at timestamp %d",
                        partition,
                        topic_name,
                        target_offset,
                        target_timestamp_ms,
                    )
                else:
                    # 没有找到对应时间戳的offset，使用最新offset
                    logger.warning(
                        "No offset found for partition %s (topic: %s) at timestamp %d, using latest offset",
                        partition,
                        topic_name,
                        target_timestamp_ms,
                    )

                    # 获取该分区的最新offset
                    latest_offset_map = await consumer.end_offsets([partition])
                    latest_offset = latest_offset_map[partition]
                    consumer.seek(partition, latest_offset)
                    seek_count += 1
                    topic_stats[topic_name]['used_latest'] += 1
                    logger.info(
                        "Seeked partition %s (topic: %s) to latest offset %d",
                        partition,
                        topic_name,
                        latest_offset,
                    )

            # 记录各topic的处理统计
            for topic_name, stats in topic_stats.items():
                logger.info(
                    "Topic '%s': %d partitions total, %d found timestamp offsets, %d used latest offsets",
                    topic_name,
                    stats['total_partitions'],
                    stats['found_offsets'],
                    stats['used_latest'],
                )

            if seek_count > 0:
                logger.info(
                    "Successfully seeked %d partitions to datetime %s",
                    seek_count,
                    offset_datetime,
                )
                return True
            else:
                logger.warning(
                    "No partitions were seeked for datetime %s", offset_datetime
                )
                return False

        except ValueError as e:
            logger.error("Invalid datetime format '%s': %s", offset_datetime, e)
            raise ValueError(
                f"Invalid datetime format '{offset_datetime}'. Expected format: 'YYYY-MM-DD HH:MM:SS'"
            ) from e

        except Exception as e:
            logger.error(
                "Failed to seek consumer to datetime %s: %s", offset_datetime, e
            )
            raise RuntimeError(
                f"Failed to seek consumer to datetime {offset_datetime}"
            ) from e
