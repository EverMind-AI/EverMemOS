# -*- coding: utf-8 -*-
"""
请求状态 DTO

用于请求状态 API 的数据传输对象。
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RequestStatusResponse(BaseModel):
    """
    请求状态响应

    包含请求的详细状态信息。
    """

    success: bool = Field(..., description="查询是否成功")
    found: bool = Field(default=False, description="是否找到请求状态")
    data: Optional[Dict[str, Any]] = Field(default=None, description="请求状态数据")
    message: Optional[str] = Field(default=None, description="提示信息")


class RequestStatusData(BaseModel):
    """
    请求状态数据模型

    Redis 中存储的请求状态信息。
    """

    organization_id: str = Field(..., description="组织 ID")
    space_id: str = Field(..., description="空间 ID")
    request_id: str = Field(..., description="请求 ID")
    status: str = Field(..., description="请求状态（start/success/failed）")
    url: Optional[str] = Field(default=None, description="请求 URL")
    method: Optional[str] = Field(default=None, description="HTTP 方法")
    http_code: Optional[int] = Field(default=None, description="HTTP 状态码")
    time_ms: Optional[int] = Field(default=None, description="请求耗时（毫秒）")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    start_time: Optional[int] = Field(default=None, description="开始时间戳（毫秒）")
    end_time: Optional[int] = Field(default=None, description="结束时间戳（毫秒）")
    ttl_seconds: Optional[int] = Field(default=None, description="剩余 TTL（秒）")

