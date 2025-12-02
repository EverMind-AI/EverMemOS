"""
租户缓存工具函数

提供租户感知的缓存机制，用于在租户上下文中缓存计算结果。
这是一个通用的缓存模式实现，避免在不同模块中重复编写相同的缓存逻辑。

核心设计模式：
- 缓存模式（Cache Pattern）
- 延迟初始化模式（Lazy Initialization Pattern）
- 备忘录模式（Memoization Pattern）

使用场景：
- 租户感知的连接别名计算
- 租户感知的数据库名称获取
- 租户感知的配置信息获取
- 任何需要按租户缓存的计算结果
"""

from typing import TypeVar, Callable, Optional, Union
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_models import TenantPatchKey

logger = get_logger(__name__)

T = TypeVar("T")


def get_or_compute_tenant_cache(
    patch_key: TenantPatchKey,
    compute_func: Callable[[], T],
    fallback: Optional[Union[T, Callable[[], T]]] = None,
    cache_description: str = "值",
) -> T:
    """
    获取或计算租户缓存值（支持延迟计算 fallback）

    这是一个通用的租户感知缓存函数，封装了常见的缓存模式：
    1. 判断是否租户模式 -> 如果不是，返回 fallback（延迟计算）
    2. 获取租户信息 -> 如果没有，返回 fallback（延迟计算）
    3. 检查 patch 缓存 -> 如果有，返回缓存值
    4. 调用 compute_func 计算新值 -> 缓存到 patch -> 返回新值

    性能优化：
    - fallback 支持延迟计算：只有在真正需要时才调用 fallback 函数
    - 避免在租户模式下执行不必要的 fallback 计算

    Args:
        patch_key: TenantPatchKey 枚举值，用于标识缓存项
        compute_func: 计算函数，当缓存未命中时调用。应该是无参数的 Callable
        fallback: 非租户模式或无租户上下文时的后备值（可选）
                 - 可以是一个具体的值（如 "default"）
                 - 也可以是一个无参数函数（延迟计算，如 lambda: get_default_database_name()）
        cache_description: 缓存项的描述，用于日志记录（可选，默认为"值"）

    Returns:
        T: 缓存的值或计算的值

    Raises:
        RuntimeError: 如果 fallback 为 None 且在非租户模式或无租户上下文时

    使用示例：
        >>> # 示例 1: 获取租户感知的连接别名（fallback 是具体值）
        >>> def compute_using():
        ...     milvus_config = get_tenant_milvus_config()
        ...     cache_key = get_milvus_connection_cache_key(milvus_config)
        ...     return f"tenant_{cache_key}"
        >>>
        >>> using = get_or_compute_tenant_cache(
        ...     patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
        ...     compute_func=compute_using,
        ...     fallback="default",  # 具体值，不需要延迟计算
        ...     cache_description="Milvus 连接别名"
        ... )

        >>> # 示例 2: 获取租户感知的数据库名称（fallback 是函数，延迟计算）
        >>> def compute_database_name():
        ...     mongo_config = get_tenant_mongo_config()
        ...     return mongo_config.get("database")
        >>>
        >>> db_name = get_or_compute_tenant_cache(
        ...     patch_key=TenantPatchKey.ACTUAL_DATABASE_NAME,
        ...     compute_func=compute_database_name,
        ...     fallback=lambda: get_default_database_name(),  # 延迟计算，只在需要时调用
        ...     cache_description="数据库名称"
        ... )
    """
    try:
        config = get_tenant_config()

        # 步骤 1: 非租户模式 -> 返回后备值（延迟计算）
        if config.non_tenant_mode:
            fallback_value = _resolve_fallback(fallback, cache_description)
            if fallback_value is None:
                raise RuntimeError(
                    f"非租户模式下必须提供 fallback 参数 [cache_key={patch_key.value}]"
                )
            logger.debug(
                "⚠️ 非租户模式，使用后备%s [fallback=%s]",
                cache_description,
                fallback_value,
            )
            return fallback_value

        # 步骤 2: 获取租户信息
        tenant_info = get_current_tenant()
        if not tenant_info:
            fallback_value = _resolve_fallback(fallback, cache_description)
            if fallback_value is None:
                raise RuntimeError(
                    f"租户模式下未设置租户上下文且未提供 fallback [cache_key={patch_key.value}]"
                )
            logger.debug(
                "⚠️ 租户模式下未设置租户上下文，使用后备%s [fallback=%s]",
                cache_description,
                fallback_value,
            )
            return fallback_value

        tenant_id = tenant_info.tenant_id

        # 步骤 3: 检查 patch 缓存
        cached_value = tenant_info.get_patch_value(patch_key)
        if cached_value is not None:
            logger.debug(
                "🔍 从 tenant_info_patch 缓存命中%s [tenant_id=%s, value=%s]",
                cache_description,
                tenant_id,
                cached_value,
            )
            return cached_value

        # 步骤 4: 计算新值
        computed_value = compute_func()

        # 步骤 5: 缓存到 patch
        tenant_info.set_patch_value(patch_key, computed_value)

        logger.debug(
            "✅ 为租户 [%s] 计算并缓存%s [value=%s]",
            tenant_id,
            cache_description,
            computed_value,
        )

        return computed_value

    except Exception as e:
        # 异常处理：尝试使用 fallback（延迟计算）
        fallback_value = _resolve_fallback(fallback, cache_description)
        if fallback_value is not None:
            logger.error(
                "获取租户缓存%s失败，使用后备值: %s [fallback=%s]",
                cache_description,
                e,
                fallback_value,
            )
            return fallback_value
        else:
            logger.error("获取租户缓存%s失败: %s", cache_description, e)
            raise


def _resolve_fallback(
    fallback: Optional[Union[T, Callable[[], T]]], description: str
) -> Optional[T]:
    """
    解析 fallback 值（支持延迟计算）

    Args:
        fallback: 可以是具体值或函数
        description: 描述信息，用于日志

    Returns:
        解析后的值
    """
    if fallback is None:
        return None

    # 如果是 Callable，调用它（延迟计算）
    if callable(fallback):
        try:
            return fallback()
        except Exception as e:
            logger.error("计算 fallback %s 失败: %s", description, e)
            return None

    # 否则直接返回具体值
    return fallback
