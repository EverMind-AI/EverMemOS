# -*- coding: utf-8 -*-
"""
请求日志事件类

用于上报请求信息给控制面消费，继承自 BaseEvent，支持 JSON 和 BSON 序列化/反序列化。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from core.events import BaseEvent


@dataclass
class RequestLogEvent(BaseEvent):
    """
    请求日志事件

    用于记录和上报 HTTP 请求的关键信息，供控制面进行监控和分析。
    继承自 BaseEvent，自动获得 event_id 和 created_at 字段。

    Attributes:
        request_id: 请求唯一标识符
        method: HTTP 请求方法（GET、POST、PUT、DELETE 等）
        url: 请求的 URL 地址
        http_code: HTTP 响应状态码
        time_ms: 请求处理耗时（毫秒）
        timestamp: 请求发生的时间戳（可选，Unix 时间戳毫秒）
        status: 自定义状态（如 "success"、"failed"、"timeout" 等）
        api_key: API 密钥标识（可选）
        organization_id: 组织 ID（可选）
        space_id: 空间 ID（可选）
        error_message: 错误信息（可选）
        extend: 扩展字段，用于存储额外信息（可选）
    """

    # 业务字段
    request_id: str = ""
    method: str = ""
    url: str = ""
    http_code: Optional[int] = None
    time_ms: Optional[int] = None
    timestamp: Optional[int] = None
    status: Optional[str] = None
    api_key: Optional[str] = None
    organization_id: Optional[str] = None
    space_id: Optional[str] = None
    error_message: Optional[str] = None
    extend: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls: Type['RequestLogEvent'], data: Dict[str, Any]
    ) -> 'RequestLogEvent':
        """
        从字典创建对象实例

        Args:
            data: 包含事件数据的字典

        Returns:
            RequestLogEvent: 对象实例

        Raises:
            KeyError: 缺少必要字段
            TypeError: 字段类型不正确
        """
        return cls(
            # 基类字段
            event_id=data.get("event_id", ""),
            created_at=data.get("created_at", ""),
            # 业务字段
            request_id=data.get("request_id", ""),
            method=data.get("method", ""),
            url=data.get("url", ""),
            http_code=data.get("http_code"),
            time_ms=data.get("time_ms"),
            timestamp=data.get("timestamp"),
            status=data.get("status"),
            api_key=data.get("api_key"),
            organization_id=data.get("organization_id"),
            space_id=data.get("space_id"),
            error_message=data.get("error_message"),
            extend=data.get("extend", {}),
        )

    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"RequestLogEvent("
            f"event_id={self.event_id!r}, "
            f"request_id={self.request_id!r}, "
            f"method={self.method!r}, "
            f"url={self.url!r}, "
            f"http_code={self.http_code}, "
            f"time_ms={self.time_ms}, "
            f"timestamp={self.timestamp}, "
            f"status={self.status!r}, "
            f"api_key={self.api_key!r}, "
            f"organization_id={self.organization_id!r}, "
            f"space_id={self.space_id!r}"
            f")"
        )
