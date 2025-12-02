"""
租户感知的 MongoDB 客户端代理

本模块实现了 AsyncMongoClient 和 AsyncDatabase 的租户感知代理版本。
核心功能：拦截所有方法调用，根据租户上下文动态切换到对应的真实客户端/数据库。
"""

from typing import Dict, Optional, Any
from pymongo.asynchronous.mongo_client import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.asynchronous.collection import AsyncCollection

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_models import TenantPatchKey
from common_utils.datetime_utils import timezone
from core.tenants.tenantize.oxm.mongo.config_utils import (
    get_tenant_mongo_config,
    get_mongo_client_cache_key,
    load_mongo_config_from_env,
    get_default_database_name,
)
from core.tenants.tenantize.tenant_cache_utils import get_or_compute_tenant_cache

logger = get_logger(__name__)


class TenantAwareMongoClient(AsyncMongoClient):
    """
    租户感知的 AsyncMongoClient 代理

    此类通过代理模式，拦截所有对 AsyncMongoClient 的调用，
    根据当前租户上下文动态切换到对应租户的真实 MongoDB 客户端。

    核心特性：
    1. 高效缓存：为每个租户缓存客户端实例，避免重复创建
    2. 租户隔离：不同租户使用不同的客户端连接
    3. 非租户模式支持：可通过配置禁用租户功能，回退到传统模式
    4. 默认客户端支持：在租户模式下，如果没有租户上下文，自动使用默认客户端（从环境变量读取）
    5. 类型兼容：通过虚拟子类注册，确保与 pymongo 和 beanie 的类型检查兼容

    使用示例：
        >>> # 租户模式（从租户上下文读取配置）
        >>> client = TenantAwareMongoClient()
        >>> db = client["my_database"]

        >>> # 租户模式下无租户上下文（使用默认客户端）
        >>> # 会从环境变量 MONGODB_* 读取默认配置
        >>> client = TenantAwareMongoClient()
        >>> db = client["my_database"]  # 使用默认客户端

        >>> # 非租户模式（使用传统参数）
        >>> client = TenantAwareMongoClient(
        ...     host="localhost",
        ...     port=27017,
        ...     username="admin",
        ...     password="password"
        ... )
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化租户感知客户端

        Args:
            host: MongoDB 主机地址（仅在非租户模式下使用）
            port: MongoDB 端口（仅在非租户模式下使用）
            username: 用户名（仅在非租户模式下使用）
            password: 密码（仅在非租户模式下使用）
            **kwargs: 其他 MongoDB 客户端参数

        缓存设计：
            - self._client_cache: 真正存储客户端实例的地方（主体缓存）
            - tenant_info_patch: 存储快捷引用（cache_key），用于快速定位应该使用哪个缓存客户端
        """
        # 客户端缓存：基于连接参数（host/port/username/password）
        # 这是真正存储客户端实例的主体缓存
        # 相同配置的不同租户可以复用同一个客户端实例
        # {cache_key: AsyncMongoClient}
        self._client_cache: Dict[str, AsyncMongoClient] = {}

        # 后备客户端（fallback client）
        # 用途：
        # 1. 非租户模式时使用（配置来自构造参数）
        # 2. 租户模式下无租户上下文时使用（配置来自环境变量）
        # 注意：一个实例只会处于一种模式，所以这两种情况不会同时发生
        self._fallback_client: Optional[AsyncMongoClient] = None

        # 后备客户端的配置
        # 优先使用构造参数，如果没有则从环境变量读取
        self._fallback_config: Optional[Dict[str, Any]] = None
        if host or port or username or password:
            # 构造参数优先（用于非租户模式）
            self._fallback_config = {
                "host": host or "localhost",
                "port": port or 27017,
                "username": username,
                "password": password,
                **kwargs,
            }

        # 配置对象
        self._config = get_tenant_config()

    def get_real_client(self) -> AsyncMongoClient:
        """
        获取真实的 MongoDB 客户端（公开方法）

        根据配置和上下文决定返回哪个客户端：
        1. 如果启用非租户模式，返回传统客户端
        2. 如果启用租户模式，根据当前租户上下文返回对应的租户客户端
        3. 如果租户模式下没有租户上下文，返回默认客户端（从环境变量读取）

        优化策略：
        - 主体缓存：self._client_cache 存储真正的客户端实例（基于连接参数）
        - 快捷引用：tenant_info_patch 存储客户端引用，用于快速访问
        - 相同连接配置的不同租户会复用同一个客户端实例

        注意：创建 AsyncMongoClient 对象本身是同步的，只是后续调用它的方法才是异步的。

        Returns:
            AsyncMongoClient: 真实的 MongoDB 客户端实例

        Raises:
            RuntimeError: 在非租户模式下但未提供连接参数，或租户配置缺失
        """

        def compute_client() -> AsyncMongoClient:
            """计算（获取或创建）租户的 MongoDB 客户端"""
            # 从租户配置中获取 MongoDB 配置
            mongo_config = get_tenant_mongo_config()
            if not mongo_config:
                tenant_info = get_current_tenant()
                raise RuntimeError(
                    f"租户 {tenant_info.tenant_id} 缺少 MongoDB 配置信息。"
                    f"请确保租户信息中包含 storage_info.mongodb 配置。"
                )

            # 基于连接参数生成缓存键
            cache_key = get_mongo_client_cache_key(mongo_config)

            # 从主体缓存获取
            if cache_key in self._client_cache:
                logger.debug("🔍 主体缓存命中 [cache_key=%s]", cache_key)
                return self._client_cache[cache_key]

            # 双重检查（防止并发创建）
            if cache_key in self._client_cache:
                return self._client_cache[cache_key]

            # 创建新的客户端
            logger.info("🔧 创建 MongoDB 客户端 [cache_key=%s]", cache_key)
            client = self._create_client_from_config(mongo_config)

            # 缓存到主体缓存
            self._client_cache[cache_key] = client
            logger.info("✅ MongoDB 客户端已缓存 [cache_key=%s]", cache_key)

            return client

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.MONGO_CLIENT_CACHE_KEY,
            compute_func=compute_client,
            fallback=lambda: self._get_fallback_client(),
            cache_description="MongoDB 客户端",
        )

    def _get_fallback_client(self) -> AsyncMongoClient:
        """
        获取后备客户端（Fallback Client）

        后备客户端用于两种场景：
        1. 非租户模式：使用构造参数提供的配置
        2. 租户模式下无租户上下文：使用环境变量的配置

        配置优先级：
        - 如果有构造参数配置（self._fallback_config），使用它
        - 否则从环境变量加载配置

        Returns:
            AsyncMongoClient: 后备客户端实例

        Raises:
            RuntimeError: 非租户模式下未提供连接参数，且无法从环境变量读取
        """
        # 检查缓存
        if self._fallback_client is not None:
            return self._fallback_client

        # 获取后备配置
        if self._fallback_config is None:
            # 没有构造参数配置，从环境变量加载
            self._fallback_config = load_mongo_config_from_env()

        # 非租户模式下，如果没有配置则报错
        if self._config.non_tenant_mode and not self._fallback_config:
            raise RuntimeError(
                "非租户模式下必须提供连接参数。"
                "请在创建 TenantAwareMongoClient 时传入 host、port 等参数，"
                "或设置环境变量 MONGODB_* 配置。"
            )

        # 创建后备客户端
        logger.info("🔧 创建后备 MongoDB 客户端")
        self._fallback_client = self._create_client_from_config(self._fallback_config)
        logger.info("✅ 后备 MongoDB 客户端已创建")

        return self._fallback_client

    def _create_client_from_config(self, config: Dict[str, Any]) -> AsyncMongoClient:
        """
        根据配置创建 MongoDB 客户端

        Args:
            config: 包含 host、port、username、password 或 uri 等字段的配置字典

        Returns:
            AsyncMongoClient: 创建的客户端实例
        """
        # 构建连接参数（包括时区和超时配置）
        conn_kwargs = {
            "serverSelectionTimeoutMS": 10000,  # PyMongo AsyncMongoClient 需要更长的超时时间
            "connectTimeoutMS": 10000,  # 连接超时
            "socketTimeoutMS": 10000,  # socket 超时
            "maxPoolSize": 50,
            "minPoolSize": 5,
            "tz_aware": True,  # 启用时区感知
            "tzinfo": timezone,  # 设置时区信息
        }

        # 优先使用 uri（如果提供）
        uri = config.get("uri")
        if uri:
            # 合并额外参数（排除 uri 和 database）
            extra_kwargs = {
                k: v for k, v in config.items() if k not in ("uri", "database")
            }
            # 用户提供的参数优先级更高
            conn_kwargs.update(extra_kwargs)
            return AsyncMongoClient(uri, **conn_kwargs)

        # 构建连接参数
        host = config.get("host", "localhost")
        port = config.get("port", 27017)
        username = config.get("username")
        password = config.get("password")

        # 构建连接字符串
        if username and password:
            from urllib.parse import quote_plus

            encoded_username = quote_plus(username)
            encoded_password = quote_plus(password)
            connection_string = (
                f"mongodb://{encoded_username}:{encoded_password}@{host}:{port}"
            )
        else:
            connection_string = f"mongodb://{host}:{port}"

        # 合并额外参数
        extra_kwargs = {
            k: v
            for k, v in config.items()
            if k not in ("host", "port", "username", "password", "database")
        }
        # 用户提供的参数优先级更高
        conn_kwargs.update(extra_kwargs)

        # 创建客户端
        return AsyncMongoClient(connection_string, **conn_kwargs)

    def __getitem__(self, key: str) -> "TenantAwareDatabase":
        """
        支持字典式访问数据库

        返回一个租户感知的 TenantAwareDatabase 对象。
        数据库名称会根据租户配置动态确定，key 参数仅作为后备值。

        Args:
            key: 请求的数据库名称（仅在租户配置未指定数据库时作为后备）

        Returns:
            TenantAwareDatabase: 租户感知的 MongoDB Database 对象
        """
        # 返回租户感知的数据库对象（不传递 key，因为会动态获取）
        return TenantAwareDatabase(self)

    def __getattr__(self, name: str) -> Any:
        """
        拦截属性访问（兜底机制）

        当属性找不到时才调用此方法，用于代理到真实的 MongoDB 客户端。
        这样可以方便地覆盖特定方法而不影响代理功能。

        Args:
            name: 属性名称

        Returns:
            Any: 代理属性或方法
        """
        # 获取真实客户端（同步）
        real_client = self.get_real_client()
        # 直接返回真实客户端的属性或方法
        return getattr(real_client, name)

    async def close(self):
        """
        关闭所有客户端连接

        清理所有缓存的客户端（主体缓存）和后备客户端。

        注意：
        - 主体缓存 self._client_cache 存储真正的客户端实例，需要管理生命周期
        - tenant_info_patch 只存储快捷引用（cache_key），不需要清理
        """
        # 关闭所有缓存的客户端（主体缓存）
        for cache_key, client in self._client_cache.items():
            try:
                await client.close()
                logger.info("🔌 已关闭 MongoDB 客户端 [cache_key=%s]", cache_key)
            except Exception as e:
                logger.error("❌ 关闭客户端失败 [cache_key=%s]: %s", cache_key, e)

        self._client_cache.clear()

        # 关闭后备客户端
        if self._fallback_client:
            try:
                await self._fallback_client.close()
                logger.info("🔌 已关闭后备 MongoDB 客户端")
            except Exception as e:
                logger.error("❌ 关闭后备客户端失败: %s", e)

            self._fallback_client = None


class TenantAwareDatabase(AsyncDatabase):
    """
    租户感知的 AsyncDatabase 代理

    此类通过代理模式，拦截所有对 AsyncDatabase 的调用，
    根据当前租户上下文动态切换到对应租户的真实数据库对象。

    核心特性：
    1. 租户隔离：不同租户使用不同的数据库实例
    2. 透明代理：所有数据库操作都会自动路由到正确的租户数据库
    3. 动态数据库名称：数据库名称根据租户上下文动态获取
    4. 类型兼容：继承 AsyncDatabase，确保与 pymongo 和 beanie 的类型检查兼容

    使用示例：
        >>> client = TenantAwareMongoClient()
        >>> db = client["my_database"]  # 返回 TenantAwareDatabase
        >>> collection = db["my_collection"]  # 自动路由到正确的租户
        >>> # 在不同租户上下文中，db.name 会返回不同的数据库名称
    """

    def __init__(self, client: TenantAwareMongoClient):
        """
        初始化租户感知数据库

        Args:
            client: 租户感知的 MongoDB 客户端

        注意：
            - 不存储数据库名称，每次访问时动态获取
            - 这样可以确保在不同租户上下文中使用正确的数据库
        """
        # 保存客户端引用
        # 注意：不调用父类 __init__，因为我们要完全代理行为
        self._tenant_aware_client = client

    def _get_real_database(self) -> AsyncDatabase:
        """
        获取真实的 MongoDB 数据库对象（带缓存）

        通过租户感知客户端获取真实的客户端，然后访问对应的数据库。
        数据库名称会根据租户配置动态获取，确保每个租户使用正确的数据库。

        优化：数据库对象会缓存在 tenant_info_patch 中，避免重复创建

        注意：一个租户只有一个数据库配置，所以使用固定的 patch_key

        Returns:
            AsyncDatabase: 真实的 MongoDB Database 对象
        """

        def compute_database() -> AsyncDatabase:
            """计算数据库对象"""
            actual_database_name = self._get_actual_database_name()
            real_client = self._tenant_aware_client.get_real_client()
            return real_client[actual_database_name]

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.MONGO_REAL_DATABASE,
            compute_func=compute_database,
            fallback=compute_database,  # fallback 逻辑相同，直接复用
            cache_description="MongoDB 数据库对象",
        )

    def _get_actual_database_name(self) -> str:
        """
        获取实际的数据库名称（动态获取，带缓存）

        根据当前租户配置动态获取真实的数据库名称：
        1. 如果在租户模式下，从租户配置中读取数据库名称（必须指定，不回退）
        2. 如果在非租户模式下，从环境变量读取默认数据库名称
        3. 如果无租户上下文，从环境变量读取默认数据库名称

        优化：数据库名称会缓存在 tenant_info_patch 中，避免重复计算

        Returns:
            str: 实际的数据库名称

        Raises:
            RuntimeError: 租户模式下如果租户配置缺失或未指定数据库名称
        """

        def compute_database_name() -> str:
            """计算数据库名称"""
            # 使用公共函数获取租户 MongoDB 配置
            mongo_config = get_tenant_mongo_config()
            if not mongo_config:
                tenant_info = get_current_tenant()
                raise RuntimeError(
                    f"租户 {tenant_info.tenant_id} 缺少 MongoDB 配置信息。"
                    f"请确保租户信息中包含 storage_info.mongodb 配置。"
                )

            # 从配置中获取数据库名称
            database_name = mongo_config.get("database")
            if not database_name:
                # 租户模式下必须指定数据库名称，不能回退到默认值
                tenant_info = get_current_tenant()
                raise RuntimeError(
                    f"租户 {tenant_info.tenant_id} 的 MongoDB 配置中未指定数据库名称。"
                    f"请在租户配置的 storage_info.mongodb.database 中指定数据库名称。"
                )

            return database_name

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.ACTUAL_DATABASE_NAME,
            compute_func=compute_database_name,
            fallback=lambda: get_default_database_name(),  # 延迟计算，只在需要时调用
            cache_description="数据库名称",
        )

    def __getitem__(self, key: str) -> AsyncCollection:
        """
        支持字典式访问集合

        Args:
            key: 集合名称

        Returns:
            AsyncCollection: MongoDB Collection 对象
        """
        # 获取真实数据库，然后访问集合
        return AsyncCollection(self, key)

    def __getattr__(self, name: str) -> Any:
        """
        拦截属性访问（兜底机制）

        当属性找不到时才调用此方法，用于代理到真实的 MongoDB 数据库对象。

        Args:
            name: 属性名称

        Returns:
            Any: 代理属性或方法
        """
        # 获取真实数据库
        real_database = self._get_real_database()
        logger.debug("🔍 获取真实的 MongoDB 数据库对象属性或方法: %s", name)
        # 直接返回真实数据库的属性或方法
        return getattr(real_database, name)

    @property
    def name(self) -> str:
        """
        获取数据库名称（动态获取）

        根据当前租户上下文动态返回实际的数据库名称。
        在不同的租户上下文中，同一个 TenantAwareDatabase 对象的 name 可能不同。

        Returns:
            str: 实际的数据库名称
        """
        return self._get_actual_database_name()

    @property
    def _name(self) -> str:
        """
        获取数据库名称

        Returns:
            str: 数据库名称
        """
        return self._get_actual_database_name()

    @property
    def client(self) -> AsyncMongoClient:
        """
        获取客户端引用（返回真实客户端）

        由于 TenantAwareDatabase 已经处于特定租户的上下文中，
        直接返回真实的 MongoDB 客户端，避免不必要的二次代理。

        Returns:
            AsyncMongoClient: 真实的 MongoDB 客户端
        """
        return self._tenant_aware_client.get_real_client()

    def __bool__(self) -> bool:
        """
        数据库对象的布尔值判断

        Returns:
            bool: 始终返回 True（数据库对象始终为真）
        """
        return True

    def __repr__(self) -> str:
        """
        数据库对象的字符串表示

        Returns:
            str: 数据库对象的描述
        """
        return (
            f"TenantAwareDatabase(client={self._tenant_aware_client}, name={self.name})"
        )

    def __eq__(self, other: Any) -> bool:
        """
        数据库对象的相等性判断

        只比较客户端引用，因为数据库名称是动态的。

        Args:
            other: 要比较的对象

        Returns:
            bool: 是否相等
        """
        if isinstance(other, TenantAwareDatabase):
            return self._tenant_aware_client == other._tenant_aware_client
        return False

    def __hash__(self) -> int:
        """
        数据库对象的哈希值

        只基于客户端引用生成哈希值。

        Returns:
            int: 哈希值
        """
        return hash(id(self._tenant_aware_client))
