"""
租户上下文管理模块

本模块提供了基于 contextvars 的租户上下文管理功能，
用于在异步环境中安全地存储和访问当前租户信息。
"""

from contextvars import ContextVar
from typing import Optional

from core.tenants.tenant_models import TenantInfo
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_info_provider import TenantInfoProvider
from core.di.container import get_container
from core.di.exceptions import BeanNotFoundError


# 全局的租户上下文变量
# 使用 ContextVar 确保在异步环境中每个任务都有独立的租户上下文
current_tenant_contextvar: ContextVar[Optional[TenantInfo]] = ContextVar(
    'current_tenant', default=None
)


def set_current_tenant(tenant_info: Optional[TenantInfo]) -> None:
    """
    设置当前请求的租户信息

    此方法将租户信息设置到当前上下文中。在异步环境下，
    每个请求/任务都有独立的上下文，互不干扰。

    Args:
        tenant_info: 租户信息对象，如果为 None 则清除当前租户信息

    Examples:
        >>> from src.core.tenants.tenant_models import TenantInfo, TenantDetail
        >>> tenant = TenantInfo(
        ...     tenant_id="tenant_001",
        ...     tenant_detail=TenantDetail(),
        ...     origin_tenant_data={}
        ... )
        >>> set_current_tenant(tenant)
    """
    current_tenant_contextvar.set(tenant_info)


def get_current_tenant() -> Optional[TenantInfo]:
    """
    获取当前请求的租户信息

    此方法从当前上下文中获取租户信息。如果当前上下文中没有设置租户信息，
    会尝试从租户配置中获取单租户ID，并通过租户信息提供者获取租户信息。

    获取逻辑：
    1. 首先从 contextvar 中获取租户信息
    2. 如果 contextvar 中没有，检查配置中是否设置了 SINGLE_TENANT_ID
    3. 如果配置了 SINGLE_TENANT_ID，通过DI容器获取 TenantInfoProvider 并获取租户信息

    Returns:
        当前上下文中的租户信息，如果未设置则返回 None

    Examples:
        >>> tenant = get_current_tenant()
        >>> if tenant:
        ...     print(f"当前租户ID: {tenant.tenant_id}")
        ... else:
        ...     print("未设置租户信息")
    """
    # 1. 首先尝试从 contextvar 中获取
    tenant_info = current_tenant_contextvar.get()
    if tenant_info is not None:
        return tenant_info

    # 2. 如果 contextvar 中没有，尝试从配置中获取 single_tenant_id
    tenant_config = get_tenant_config()
    single_tenant_id = tenant_config.single_tenant_id

    # 3. 如果配置了 single_tenant_id，通过DI容器获取 TenantInfoProvider
    if single_tenant_id:
        try:
            # 从DI容器获取 TenantInfoProvider 实例（会自动选择 primary 实现）
            provider = get_container().get_bean_by_type(TenantInfoProvider)
            tenant_info = provider.get_tenant_info(single_tenant_id)
            set_current_tenant(tenant_info)
            return tenant_info
        except BeanNotFoundError:
            # 如果DI容器中没有注册 TenantInfoProvider，返回 None
            # 这种情况通常出现在应用启动早期或测试环境
            return None

    # 4. 如果都没有，返回 None
    return None


def clear_current_tenant() -> None:
    """
    清除当前请求的租户信息

    此方法将当前上下文中的租户信息设置为 None，
    相当于 set_current_tenant(None)。

    Examples:
        >>> clear_current_tenant()
    """
    current_tenant_contextvar.set(None)


def get_current_tenant_id() -> Optional[str]:
    """
    获取当前租户的 ID

    这是一个便捷方法，直接返回当前租户的 tenant_id。
    如果当前没有设置租户信息，则返回 None。

    Returns:
        当前租户的 ID，如果未设置租户信息则返回 None

    Examples:
        >>> tenant_id = get_current_tenant_id()
        >>> if tenant_id:
        ...     print(f"当前租户ID: {tenant_id}")
    """
    tenant = get_current_tenant()
    return tenant.tenant_id if tenant else None
