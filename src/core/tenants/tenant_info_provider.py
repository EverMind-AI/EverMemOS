"""
租户信息提供者模块

本模块定义了租户信息提供者接口及其默认实现，
用于根据 tenant_id 获取租户信息。

使用DI机制管理 TenantInfoProvider 的实现。
"""

from abc import ABC, abstractmethod
from typing import Optional

from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.di.decorators import component


class TenantInfoProvider(ABC):
    """
    租户信息提供者接口

    此接口定义了获取租户信息的标准方法。
    不同的实现可以从不同的数据源获取租户信息（如数据库、API、配置文件等）。

    使用DI机制：
    - 可以注册多个实现
    - 使用 primary=True 标记默认实现
    - 通过容器获取实例
    """

    @abstractmethod
    def get_tenant_info(self, tenant_id: str) -> Optional[TenantInfo]:
        """
        根据租户ID获取租户信息

        Args:
            tenant_id: 租户唯一标识符

        Returns:
            租户信息对象，如果未找到则返回 None
        """
        raise NotImplementedError


@component("default_tenant_info_provider")
class DefaultTenantInfoProvider(TenantInfoProvider):
    """
    默认的租户信息提供者实现

    此实现提供最基本的租户信息，仅包含 tenant_id，
    不包含存储配置等详细信息。适用于简单场景或作为默认实现。

    使用 @component 装饰器注册到DI容器，并标记为 primary。
    """

    def get_tenant_info(self, tenant_id: str) -> Optional[TenantInfo]:
        """
        根据租户ID创建基本的租户信息对象

        此实现创建一个仅包含 tenant_id 的 TenantInfo 对象，
        不包含任何存储配置信息。

        Args:
            tenant_id: 租户唯一标识符

        Returns:
            包含基本信息的 TenantInfo 对象

        Examples:
            >>> from core.di.container import get_container
            >>> provider = get_container().get_bean_by_type(TenantInfoProvider)
            >>> tenant_info = provider.get_tenant_info("tenant_001")
            >>> print(tenant_info.tenant_id)
            tenant_001
        """
        # 如果 tenant_id 为空，返回 None
        if not tenant_id:
            return None

        # 创建仅包含 tenant_id 的基本租户信息
        return TenantInfo(
            tenant_id=tenant_id, tenant_detail=TenantDetail(), origin_tenant_data={}
        )
