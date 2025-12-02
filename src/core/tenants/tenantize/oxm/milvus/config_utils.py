"""
Milvus 租户配置工具函数

本模块提供租户 Milvus 配置相关的工具函数，用于从租户信息中提取 Milvus 配置。
"""

import os
from typing import Optional, Dict, Any
from hashlib import md5

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_config import get_tenant_config

logger = get_logger(__name__)


def get_tenant_milvus_config() -> Optional[Dict[str, Any]]:
    """
    从当前租户上下文中获取 Milvus 配置

    从租户信息的 storage_info 中提取 Milvus 相关配置。
    如果租户配置不完整或缺失，会从环境变量中补充。

    配置结构示例：
    {
        "host": "localhost",
        "port": 19530,
        "user": "admin",
        "password": "password"
    }

    注意：
        Milvus 的租户隔离通过 Collection 名称实现（在 TenantAwareCollection 中处理），
        不使用 db_name 维度的隔离。

    Returns:
        Milvus 配置字典，如果不存在则返回 None

    Examples:
        >>> config = get_tenant_milvus_config()
        >>> if config:
        ...     print(f"Milvus URI: {config['host']}:{config['port']}")
    """
    tenant_info = get_current_tenant()
    if not tenant_info:
        logger.debug("⚠️ 未设置租户上下文，无法获取租户 Milvus 配置")
        return None

    # 从租户的 storage_info 中获取 Milvus 配置
    # 支持两种配置键名：milvus 或 milvus_config
    milvus_config = tenant_info.get_storage_info("milvus")
    if milvus_config is None:
        milvus_config = tenant_info.get_storage_info("milvus_config")

    # 获取环境变量配置作为后备
    env_fallback_config = load_milvus_config_from_env()

    if not milvus_config:
        # 租户配置中完全没有 Milvus 信息，使用环境变量配置
        final_config = {
            "host": env_fallback_config.get("host", "localhost"),
            "port": env_fallback_config.get("port", 19530),
            "user": env_fallback_config.get("user", ""),
            "password": env_fallback_config.get("password", ""),
        }
        logger.info(
            "✅ 租户 [%s] 配置中缺少 Milvus 信息，使用环境变量配置补全: host=%s, port=%s",
            tenant_info.tenant_id,
            final_config["host"],
            final_config["port"],
        )
        return final_config

    # 兼容逻辑：如果租户配置缺少某些字段，从环境变量中补充
    final_config = {
        "host": milvus_config.get("host")
        or env_fallback_config.get("host", "localhost"),
        "port": milvus_config.get("port") or env_fallback_config.get("port", 19530),
        "user": milvus_config.get("user") or env_fallback_config.get("user", ""),
        "password": milvus_config.get("password")
        or env_fallback_config.get("password", ""),
    }

    logger.debug(
        "✅ 从租户 [%s] 获取 Milvus 配置: host=%s, port=%s",
        tenant_info.tenant_id,
        final_config["host"],
        final_config["port"],
    )

    return final_config


def get_milvus_connection_cache_key(config: Dict[str, Any]) -> str:
    """
    基于 Milvus 连接配置生成缓存键

    使用连接参数（host、port、user、password）的哈希值作为缓存键。
    注意：db_name 不参与缓存键生成，因为同一个连接可以访问不同的数据库。

    Args:
        config: Milvus 连接配置字典

    Returns:
        缓存键字符串（MD5 哈希值）

    Examples:
        >>> config = {"host": "localhost", "port": 19530, "user": "admin", "password": "pwd"}
        >>> cache_key = get_milvus_connection_cache_key(config)
    """
    # 使用连接参数生成唯一的缓存键（不包括 db_name，因为同一连接可以访问多个数据库）
    key_parts = [
        str(config.get("host", "")),
        str(config.get("port", "")),
        str(config.get("user", "")),
        str(config.get("password", "")),
    ]
    key_str = "|".join(key_parts)
    cache_key = md5(key_str.encode()).hexdigest()[:16]
    return cache_key


def load_milvus_config_from_env() -> Dict[str, Any]:
    """
    从环境变量加载默认 Milvus 配置

    读取以下环境变量：
    - MILVUS_HOST: Milvus 主机地址，默认 localhost
    - MILVUS_PORT: Milvus 端口，默认 19530
    - MILVUS_USER: 用户名（可选）
    - MILVUS_PASSWORD: 密码（可选）

    注意：
        不使用 MILVUS_DB_NAME，租户隔离通过 Collection 名称实现

    Returns:
        Milvus 配置字典

    Examples:
        >>> config = load_milvus_config_from_env()
        >>> print(f"Milvus URI: {config['host']}:{config['port']}")
    """
    config = {
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": int(os.getenv("MILVUS_PORT", "19530")),
        "user": os.getenv("MILVUS_USER", ""),
        "password": os.getenv("MILVUS_PASSWORD", ""),
    }

    logger.debug(
        "从环境变量加载默认 Milvus 配置: host=%s, port=%s",
        config["host"],
        config["port"],
    )

    return config


def get_tenant_aware_collection_name(original_name: str) -> str:
    """
    生成租户感知的 Collection 名称

    根据当前租户上下文为 Collection 名称添加租户前缀。
    如果在非租户模式或无租户上下文，返回原始名称。

    命名规则：
    - 添加租户前缀：{tenant_id}_{original_name}
    - 替换特殊字符：将 "-" 和 "." 替换为 "_" 以符合 Milvus 命名规范

    Args:
        original_name: 原始的 Collection 名称

    Returns:
        str: 租户感知的 Collection 名称

    Examples:
        >>> # 租户模式下
        >>> set_current_tenant(TenantInfo(tenant_id="tenant-001", ...))
        >>> get_tenant_aware_collection_name("my_collection")
        'tenant_001_my_collection'

        >>> # 非租户模式或无租户上下文
        >>> get_tenant_aware_collection_name("my_collection")
        'my_collection'
    """
    try:

        # 检查是否为非租户模式
        config = get_tenant_config()
        if config.non_tenant_mode:
            return original_name

        # 获取当前租户信息
        tenant_info = get_current_tenant()
        if not tenant_info:
            return original_name

        # 生成租户前缀（替换特殊字符以符合 Milvus 命名规范）
        tenant_prefix = tenant_info.tenant_id.replace("-", "_").replace(".", "_")

        # 返回租户感知的表名
        return f"{tenant_prefix}_{original_name}"

    except Exception as e:
        logger.warning(
            "生成租户感知的 Collection 名称失败，使用原始名称 [%s]: %s",
            original_name,
            e,
        )
        return original_name
