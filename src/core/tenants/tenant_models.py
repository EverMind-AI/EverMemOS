"""
租户信息数据模型

本模块定义了租户相关的数据类，用于统一管理租户信息。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class TenantPatchKey(str, Enum):
    """
    租户信息 Patch 缓存键枚举

    用于统一管理 tenant_info_patch 中的缓存键，避免字符串硬编码导致的键名冲突。

    设计说明：
    - 所有 patch 缓存键都应该在这里定义
    - 使用枚举可以提供更好的类型提示和防止拼写错误
    - 继承自 str，使其可以直接作为字典 key 使用

    缓存键说明：
    - MONGO_CLIENT_CACHE_KEY: 用于缓存 MongoDB 客户端的 cache_key（连接参数的哈希值）
    - ACTUAL_DATABASE_NAME: 用于缓存实际使用的数据库名称
    - REAL_DATABASE_PREFIX: 用于缓存数据库对象的键前缀（实际使用时需要拼接数据库名）
    """

    # MongoDB 客户端相关
    MONGO_CLIENT_CACHE_KEY = "mongo_client_cache_key"

    # MongoDB 数据库相关
    ACTUAL_DATABASE_NAME = "actual_database_name"
    MONGO_REAL_DATABASE = "mongo_real_database"  # 真实的 MongoDB 数据库对象

    # Milvus 连接相关
    MILVUS_CONNECTION_CACHE_KEY = "milvus_connection_cache_key"


@dataclass
class TenantDetail:
    """
    租户详细信息数据类

    这个类用于存储经过适配后的租户详细信息，外部的租户信息会被适配成这个数据模型。

    Attributes:
        storage_info: 存储相关配置信息的字典，可选字段
                     结构示例: {
                         "mongodb": {"host": "...", "port": 27017, ...},
                         "redis": {"host": "...", "port": 6379, ...}
                     }
    """

    storage_info: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class TenantInfo:
    """
    租户信息数据类

    这个类是租户信息的主要数据模型，包含租户的核心信息。

    Attributes:
        tenant_id: 租户唯一标识符
        tenant_detail: 经过适配后的租户详细信息
        origin_tenant_data: 外部直接传入的原始租户数据，保持原样不做适配
        tenant_info_patch: 租户相关的缓存数据，用于存储计算后的值（如 actual_database_name, real_client 等）
                          生命周期与 tenant_info 一致，避免重复计算
    """

    tenant_id: str
    tenant_detail: TenantDetail
    origin_tenant_data: Dict[str, Any] = field(default_factory=dict)
    tenant_info_patch: Dict[str, Any] = field(default_factory=dict)

    def get_storage_info(self, storage_type: str) -> Optional[Dict[str, Any]]:
        """
        获取指定类型的存储配置信息

        此方法从 tenant_detail.storage_info 中获取指定类型的存储配置。

        Args:
            storage_type: 存储类型，如 "mongodb", "redis", "elasticsearch" 等

        Returns:
            指定存储类型的配置信息字典，如果不存在则返回 None

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(storage_info={
            ...         "mongodb": {"host": "localhost", "port": 27017}
            ...     }),
            ...     origin_tenant_data={}
            ... )
            >>> mongo_config = tenant_info.get_storage_info("mongodb")
            >>> print(mongo_config)
            {'host': 'localhost', 'port': 27017}
        """
        # 检查 tenant_detail.storage_info 是否存在
        if self.tenant_detail.storage_info is None:
            return None

        # 从 storage_info 中获取指定类型的配置
        return self.tenant_detail.storage_info.get(storage_type)

    def get_patch_value(self, key: str, default: Any = None) -> Any:
        """
        从 tenant_info_patch 中获取缓存值

        此方法用于获取缓存在 tenant_info_patch 中的计算值，
        避免重复计算，提高性能。

        Args:
            key: 缓存键名
            default: 如果键不存在时的默认返回值

        Returns:
            缓存的值，如果不存在则返回 default

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> # 获取缓存的数据库名称
            >>> db_name = tenant_info.get_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME)
        """
        return self.tenant_info_patch.get(key, default)

    def set_patch_value(self, key: str, value: Any) -> None:
        """
        设置 tenant_info_patch 中的缓存值

        此方法用于缓存计算结果，避免后续重复计算。
        缓存的生命周期与 TenantInfo 实例一致。

        Args:
            key: 缓存键名
            value: 要缓存的值

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> # 缓存数据库名称
            >>> tenant_info.set_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME, "tenant_001_db")
            >>> # 缓存客户端的 cache_key
            >>> tenant_info.set_patch_value(TenantPatchKey.MONGO_CLIENT_CACHE_KEY, "cache_key_value")
        """
        self.tenant_info_patch[key] = value

    def has_patch_value(self, key: str) -> bool:
        """
        检查 tenant_info_patch 中是否存在指定的缓存值

        Args:
            key: 缓存键名

        Returns:
            如果存在返回 True，否则返回 False

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME, "tenant_001_db")
            >>> tenant_info.has_patch_value(TenantPatchKey.ACTUAL_DATABASE_NAME)
            True
            >>> tenant_info.has_patch_value("non_existent_key")
            False
        """
        return key in self.tenant_info_patch

    def clear_patch_cache(self) -> None:
        """
        清除所有缓存数据

        在某些情况下可能需要清除缓存（如租户配置更新时），
        此方法会清空 tenant_info_patch 中的所有数据。

        Examples:
            >>> tenant_info = TenantInfo(
            ...     tenant_id="tenant_001",
            ...     tenant_detail=TenantDetail(),
            ...     origin_tenant_data={}
            ... )
            >>> tenant_info.set_patch_value("key1", "value1")
            >>> tenant_info.clear_patch_cache()
            >>> tenant_info.has_patch_value("key1")
            False
        """
        self.tenant_info_patch.clear()
