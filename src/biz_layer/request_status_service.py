# -*- coding: utf-8 -*-
"""
请求状态服务

负责将请求状态写入 Redis（使用 Hash 结构），并提供读取功能。
用于跟踪转后台的请求状态。

Redis Key 格式: request_status:{organization_id}:{space_id}:{request_id}
TTL: 2 小时
"""

from typing import Any, Dict, Optional

from core.component.redis_provider import RedisProvider
from core.di import service
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Redis key 前缀
REQUEST_STATUS_KEY_PREFIX = "request_status"

# TTL: 1 小时（秒）
REQUEST_STATUS_TTL = 60 * 60


@service("request_status_service")
class RequestStatusService:
    """
    请求状态服务

    负责：
    - 将请求状态写入 Redis（使用 Hash 结构，便于扩展）
    - 提供读取特定请求状态的功能
    - 设置 2 小时的 TTL

    Redis Hash 结构示例:
    request_status:{org_id}:{space_id}:{request_id} = {
        "status": "start|success|failed",
        "url": "请求 URL",
        "method": "GET|POST|...",
        "http_code": "200",
        "time_ms": "123",
        "error_message": "错误信息（如有）",
        "start_time": "开始时间戳",
        "end_time": "结束时间戳"
    }
    """

    def __init__(self):
        """初始化服务"""
        # 延迟获取 RedisProvider，避免循环依赖
        self._redis_provider: Optional[RedisProvider] = None

    def _get_redis_provider(self) -> RedisProvider:
        """
        获取 Redis Provider（懒加载）

        Returns:
            RedisProvider: Redis 提供者实例
        """
        if self._redis_provider is None:
            self._redis_provider = get_bean_by_type(RedisProvider)
        return self._redis_provider

    def _build_key(self, organization_id: str, space_id: str, request_id: str) -> str:
        """
        构建 Redis Key

        Args:
            organization_id: 组织 ID
            space_id: 空间 ID
            request_id: 请求 ID

        Returns:
            str: Redis key
        """
        return f"{REQUEST_STATUS_KEY_PREFIX}:{organization_id}:{space_id}:{request_id}"

    async def update_request_status(
        self,
        organization_id: str,
        space_id: str,
        request_id: str,
        status: str,
        url: Optional[str] = None,
        method: Optional[str] = None,
        http_code: Optional[int] = None,
        time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> bool:
        """
        更新请求状态到 Redis

        Args:
            organization_id: 组织 ID
            space_id: 空间 ID
            request_id: 请求 ID
            status: 请求状态（start/success/failed）
            url: 请求 URL（可选）
            method: HTTP 方法（可选）
            http_code: HTTP 状态码（可选）
            time_ms: 请求耗时毫秒（可选）
            error_message: 错误信息（可选）
            timestamp: 时间戳（可选）

        Returns:
            bool: 是否更新成功
        """
        if not organization_id or not space_id or not request_id:
            logger.warning(
                "缺少必要参数，跳过请求状态更新: org=%s, space=%s, req=%s",
                organization_id,
                space_id,
                request_id,
            )
            return False

        try:
            redis_provider = self._get_redis_provider()
            client = await redis_provider.get_client()

            key = self._build_key(organization_id, space_id, request_id)

            # 构建要更新的字段
            fields: Dict[str, str] = {"status": status}

            if url is not None:
                fields["url"] = url
            if method is not None:
                fields["method"] = method
            if http_code is not None:
                fields["http_code"] = str(http_code)
            if time_ms is not None:
                fields["time_ms"] = str(time_ms)
            if error_message is not None:
                fields["error_message"] = error_message
            if timestamp is not None:
                # 根据状态设置不同的时间字段
                if status == "start":
                    fields["start_time"] = str(timestamp)
                else:
                    fields["end_time"] = str(timestamp)

            # 使用 hset 更新 hash 字段
            await client.hset(key, mapping=fields)

            # 设置/刷新 TTL
            await client.expire(key, REQUEST_STATUS_TTL)

            logger.debug("请求状态已更新到 Redis: key=%s, status=%s", key, status)
            return True

        except Exception as e:
            logger.error(
                "更新请求状态到 Redis 失败: org=%s, space=%s, req=%s, error=%s",
                organization_id,
                space_id,
                request_id,
                str(e),
            )
            return False

    async def get_request_status(
        self, organization_id: str, space_id: str, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取请求状态

        Args:
            organization_id: 组织 ID
            space_id: 空间 ID
            request_id: 请求 ID

        Returns:
            Optional[Dict[str, Any]]: 请求状态信息，如果不存在则返回 None
        """
        if not organization_id or not space_id or not request_id:
            logger.warning(
                "缺少必要参数，无法获取请求状态: org=%s, space=%s, req=%s",
                organization_id,
                space_id,
                request_id,
            )
            return None

        try:
            redis_provider = self._get_redis_provider()
            client = await redis_provider.get_client()

            key = self._build_key(organization_id, space_id, request_id)

            # 获取所有 hash 字段
            data = await client.hgetall(key)

            if not data:
                logger.debug("请求状态不存在: key=%s", key)
                return None

            # 转换数据类型
            result: Dict[str, Any] = {
                "organization_id": organization_id,
                "space_id": space_id,
                "request_id": request_id,
            }

            for field, value in data.items():
                if field in ("http_code", "time_ms", "start_time", "end_time"):
                    # 数字字段转换为 int
                    try:
                        result[field] = int(value)
                    except (ValueError, TypeError):
                        result[field] = value
                else:
                    result[field] = value

            # 获取剩余 TTL
            ttl = await client.ttl(key)
            if ttl > 0:
                result["ttl_seconds"] = ttl

            logger.debug("获取请求状态成功: key=%s", key)
            return result

        except Exception as e:
            logger.error(
                "获取请求状态失败: org=%s, space=%s, req=%s, error=%s",
                organization_id,
                space_id,
                request_id,
                str(e),
            )
            return None

    async def delete_request_status(
        self, organization_id: str, space_id: str, request_id: str
    ) -> bool:
        """
        删除请求状态

        Args:
            organization_id: 组织 ID
            space_id: 空间 ID
            request_id: 请求 ID

        Returns:
            bool: 是否删除成功
        """
        if not organization_id or not space_id or not request_id:
            return False

        try:
            redis_provider = self._get_redis_provider()
            key = self._build_key(organization_id, space_id, request_id)
            deleted = await redis_provider.delete(key)
            logger.debug("请求状态已删除: key=%s, deleted=%d", key, deleted)
            return deleted > 0

        except Exception as e:
            logger.error(
                "删除请求状态失败: org=%s, space=%s, req=%s, error=%s",
                organization_id,
                space_id,
                request_id,
                str(e),
            )
            return False

