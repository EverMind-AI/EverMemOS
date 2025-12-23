# -*- coding: utf-8 -*-
"""
请求状态控制器

提供 API 用于查询请求的处理状态。
主要用于跟踪转后台的请求。
"""

from typing import Optional

from fastapi import Header, HTTPException

from core.di.decorators import component
from core.interface.controller.base_controller import BaseController, get
from core.observation.logger import get_logger
from biz_layer.request_status_service import RequestStatusService
from infra_layer.adapters.input.api.dto.status_dto import (
    RequestStatusResponse,
)


logger = get_logger(__name__)


@component(name="statusController")
class StatusController(BaseController):
    """
    请求状态控制器

    提供 API 用于查询请求的处理状态，主要用于跟踪转后台的请求。
    """

    def __init__(self, request_status_service: RequestStatusService):
        """
        初始化控制器

        Args:
            request_status_service: 请求状态服务（通过依赖注入）
        """
        super().__init__(
            prefix="/api/v1/stats",
            tags=["Stats - Request Status"],
            default_auth="none",  # 根据需求调整认证策略
        )
        self.request_status_service = request_status_service
        logger.info("StatusController initialized")

    @get(
        "/request",
        response_model=RequestStatusResponse,
        summary="查询请求状态",
        description="""
        查询特定请求的处理状态

        ## 功能说明：
        - 根据 organization_id、space_id、request_id 查询请求状态
        - 返回请求的处理进度（start/success/failed）
        - 支持查看请求耗时、HTTP 状态码等信息

        ## 参数传递方式：
        通过 HTTP Header 传递参数（与其他 API 保持一致）：
        - X-Organization-Id: 组织 ID
        - X-Space-Id: 空间 ID
        - X-Request-Id: 请求 ID

        ## 使用场景：
        - 转后台请求的状态跟踪
        - 客户端轮询请求完成状态

        ## 注意：
        - 请求状态数据有 2 小时的 TTL，过期后将无法查询

        ## 接口路径：
        GET /api/v1/stats/request
        """,
        responses={
            200: {
                "description": "查询成功",
                "content": {
                    "application/json": {
                        "example": {
                            "success": True,
                            "found": True,
                            "data": {
                                "organization_id": "org-123",
                                "space_id": "space-456",
                                "request_id": "req-789",
                                "status": "success",
                                "url": "/api/memory/memorize",
                                "method": "POST",
                                "http_code": 200,
                                "time_ms": 1500,
                                "start_time": 1702400000000,
                                "end_time": 1702400001500,
                                "ttl_seconds": 7100,
                            },
                            "message": None,
                        }
                    }
                },
            },
            400: {
                "description": "参数错误",
                "content": {
                    "application/json": {
                        "example": {"detail": "缺少必要的 Header 参数"}
                    }
                },
            },
        },
    )
    async def get_request_status(
        self,
        x_organization_id: Optional[str] = Header(
            None, alias="X-Organization-Id", description="组织 ID"
        ),
        x_space_id: Optional[str] = Header(
            None, alias="X-Space-Id", description="空间 ID"
        ),
        x_request_id: Optional[str] = Header(
            None, alias="X-Request-Id", description="请求 ID"
        ),
    ) -> RequestStatusResponse:
        """
        查询请求状态

        通过 HTTP Header 传递参数：
        - X-Organization-Id
        - X-Space-Id
        - X-Request-Id

        Returns:
            RequestStatusResponse: 请求状态响应
        """
        # 参数校验
        if not x_organization_id or not x_space_id or not x_request_id:
            missing = []
            if not x_organization_id:
                missing.append("X-Organization-Id")
            if not x_space_id:
                missing.append("X-Space-Id")
            if not x_request_id:
                missing.append("X-Request-Id")
            raise HTTPException(
                status_code=400, detail=f"缺少必要的 Header 参数: {', '.join(missing)}"
            )

        try:
            # 查询状态
            data = await self.request_status_service.get_request_status(
                organization_id=x_organization_id,
                space_id=x_space_id,
                request_id=x_request_id,
            )

            if data is None:
                return RequestStatusResponse(
                    success=True,
                    found=False,
                    data=None,
                    message="请求状态不存在或已过期",
                )

            return RequestStatusResponse(
                success=True, found=True, data=data, message=None
            )

        except Exception as e:
            logger.error(
                "查询请求状态异常: org=%s, space=%s, req=%s, error=%s",
                x_organization_id,
                x_space_id,
                x_request_id,
                str(e),
                exc_info=True,
            )
            return RequestStatusResponse(
                success=False, found=False, data=None, message=f"查询失败: {str(e)}"
            )

