"""
MongoDB 配置工具函数

提供租户感知的 MongoDB 配置相关的公共工具函数。
"""

import os
from typing import Optional, Dict, Any
from functools import lru_cache
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_config import get_tenant_config

logger = get_logger(__name__)

DEFAULT_DATABASE_NAME = "memsys"


def get_tenant_mongo_config() -> Optional[Dict[str, Any]]:
    """
    获取当前租户的 MongoDB 配置信息

    从租户信息的 storage_info 中提取 MongoDB 相关配置。
    如果租户配置不完整或缺失，会从环境变量中补充（database 除外）。

    Returns:
        Optional[Dict[str, Any]]: MongoDB 配置字典，如果无法获取则返回 None

    配置字典可能包含的字段：
        - uri: MongoDB 连接 URI
        - host: MongoDB 主机地址
        - port: MongoDB 端口
        - username: 用户名
        - password: 密码
        - database: 数据库名称（只从租户配置获取，不从环境变量回退）
        - 其他连接参数
    """
    tenant_info = get_current_tenant()
    if not tenant_info:
        logger.debug("⚠️ 无法获取租户信息，返回 None")
        return None

    mongo_config = tenant_info.get_storage_info("mongodb")

    # 获取环境变量配置作为后备
    env_fallback_config = load_mongo_config_from_env()

    if not mongo_config:
        final_config = {
            "host": env_fallback_config.get("host", "localhost"),
            "port": env_fallback_config.get("port", 27017),
            "username": env_fallback_config.get("username"),
            "password": env_fallback_config.get("password"),
            "database": generate_tenant_database_name(DEFAULT_DATABASE_NAME),
        }
        logger.info(
            "✅ 租户 [%s] 配置中缺少 MongoDB 信息，使用环境变量配置补全: %s, database=%s",
            tenant_info.tenant_id,
            final_config.get("uri")
            or f"host={final_config.get('host')}:{final_config.get('port')}",
            final_config.get("database"),
        )
        return final_config

    # 兼容逻辑：如果租户配置缺少某些字段，从环境变量中补充（database 除外）
    # 优先使用 URI（完整连接字符串）
    if mongo_config.get("uri"):
        final_config = {
            "uri": mongo_config["uri"],
            # database：如果租户配置有指定则使用，否则生成租户感知的名称
            "database": mongo_config.get("database")
            or generate_tenant_database_name(DEFAULT_DATABASE_NAME),
        }
    else:
        # 使用独立的连接参数
        final_config = {
            "host": mongo_config.get("host")
            or env_fallback_config.get("host", "localhost"),
            "port": mongo_config.get("port") or env_fallback_config.get("port", 27017),
            "username": mongo_config.get("username")
            or env_fallback_config.get("username"),
            "password": mongo_config.get("password")
            or env_fallback_config.get("password"),
            # database：如果租户配置有指定则使用，否则生成租户感知的名称
            "database": mongo_config.get("database")
            or generate_tenant_database_name(DEFAULT_DATABASE_NAME),
        }

    logger.debug(
        "✅ 从租户 [%s] 获取 MongoDB 配置: %s, database=%s",
        tenant_info.tenant_id,
        (
            "uri"
            if final_config.get("uri")
            else f"host={final_config.get('host')}:{final_config.get('port')}"
        ),
        final_config.get("database") or "(未指定)",
    )

    return final_config


def get_mongo_client_cache_key(config: Dict[str, Any]) -> str:
    """
    根据 MongoDB 配置生成缓存键

    基于连接参数（host/port/username/password/uri）生成唯一的缓存键，
    这样相同配置的连接可以复用同一个客户端实例。

    Args:
        config: MongoDB 配置字典

    Returns:
        str: 缓存键
    """
    # 优先使用 URI 生成缓存键
    uri = config.get("uri")
    if uri:
        # 对于 URI，直接使用其作为主要标识
        # 注意：URI 中可能包含敏感信息，但这只是内存中的缓存键
        return f"uri:{uri}"

    # 使用 host/port/username 组合生成缓存键
    host = config.get("host", "localhost")
    port = config.get("port", 27017)
    username = config.get("username", "")

    # 不包含 password 在缓存键中（因为 password 相同时，其他参数也应该相同）
    # 不包含 database 在缓存键中（同一个客户端可以访问多个数据库）
    cache_key = f"host:{host}:port:{port}:user:{username}"

    return cache_key


def load_mongo_config_from_env() -> Dict[str, Any]:
    """
    从环境变量加载 MongoDB 配置

    读取 MONGODB_* 环境变量，优先使用 MONGODB_URI。
    用于后备客户端或默认客户端的配置加载。

    Returns:
        Dict[str, Any]: 包含连接信息的配置字典

    环境变量：
        - MONGODB_URI: MongoDB 连接 URI（优先使用）
        - MONGODB_HOST: MongoDB 主机地址（默认：localhost）
        - MONGODB_PORT: MongoDB 端口（默认：27017）
        - MONGODB_USERNAME: 用户名（可选）
        - MONGODB_PASSWORD: 密码（可选）
        - MONGODB_DATABASE: 数据库名称（默认：memsys）
    """
    # 优先使用 MONGODB_URI
    uri = os.getenv("MONGODB_URI")
    if uri:
        logger.info("📋 从环境变量 MONGODB_URI 加载配置")
        return {"uri": uri, "database": get_default_database_name()}

    # 分别读取各个配置项
    host = os.getenv("MONGODB_HOST", "localhost")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    database = get_default_database_name()

    logger.info(
        "📋 从环境变量加载配置: host=%s, port=%s, database=%s", host, port, database
    )

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "database": database,
    }


@lru_cache(maxsize=1)
def get_default_database_name() -> str:
    """
    获取默认的数据库名称

    从环境变量 MONGODB_DATABASE 读取，如果未设置则返回 "memsys"。

    Returns:
        str: 默认的数据库名称
    """
    return os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE_NAME)


def generate_tenant_database_name(base_name: str = "memsys") -> str:
    """
    生成租户感知的数据库名称

    根据当前租户上下文为数据库名称添加租户前缀。
    如果在非租户模式或无租户上下文，返回原始名称。

    命名规则：
    - 添加租户前缀：{tenant_id}_{base_name}
    - 替换特殊字符：将 "-" 和 "." 替换为 "_" 以符合 MongoDB 命名规范

    Args:
        base_name: 基础数据库名称，默认为 "memsys"

    Returns:
        str: 租户感知的数据库名称

    Examples:
        >>> # 租户模式下
        >>> set_current_tenant(TenantInfo(tenant_id="tenant-001", ...))
        >>> generate_tenant_database_name("memsys")
        'tenant_001_memsys'

        >>> # 非租户模式或无租户上下文
        >>> generate_tenant_database_name("memsys")
        'memsys'
    """
    try:

        # 检查是否为非租户模式
        config = get_tenant_config()
        if config.non_tenant_mode:
            return base_name

        # 获取当前租户信息
        tenant_info = get_current_tenant()
        if not tenant_info:
            return base_name

        # 生成租户前缀（替换特殊字符以符合 MongoDB 命名规范）
        tenant_prefix = tenant_info.tenant_id.replace("-", "_").replace(".", "_")

        # 返回租户感知的数据库名称
        return f"{tenant_prefix}_{base_name}"

    except Exception as e:
        logger.warning(
            "生成租户感知的数据库名称失败，使用原始名称 [%s]: %s", base_name, e
        )
        return base_name
