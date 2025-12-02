"""
租户感知的 Milvus Collection

本模块通过继承 pymilvus.Collection 并覆盖 _get_connection 方法来实现租户感知能力。
核心思路：根据租户上下文动态返回正确的连接 handler。
"""

from typing import Optional
from pymilvus import Collection, CollectionSchema
from pymilvus.orm.connections import connections

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant
from core.tenants.tenant_models import TenantPatchKey
from core.tenants.tenantize.oxm.milvus.config_utils import (
    get_tenant_milvus_config,
    get_milvus_connection_cache_key,
    load_milvus_config_from_env,
    get_tenant_aware_collection_name,
)
from core.tenants.tenantize.tenant_cache_utils import get_or_compute_tenant_cache
from component.milvus_client_factory import MilvusClientFactory
from core.di.utils import get_bean_by_type

logger = get_logger(__name__)


class TenantAwareCollection(Collection):
    """
    租户感知的 Milvus Collection

    通过继承 pymilvus.Collection 并覆盖 _get_connection 方法来实现租户感知。
    核心功能：根据当前租户上下文，自动选择并返回正确的 Milvus 连接。

    核心特性：
    1. 租户隔离：不同租户使用不同的 Milvus 连接（通过 using 别名区分）
    2. 连接复用：相同配置的租户共享同一个连接（通过 cache_key 缓存）
    3. 自动注册：首次访问时自动注册租户连接
    4. 后备连接：非租户模式或无租户上下文时使用默认连接

    使用方式：
        >>> # 在 MilvusCollectionBase 中使用
        >>> class MyCollectionBase(MilvusCollectionBase):
        ...     def load_collection(self) -> Collection:
        ...         # 使用 TenantAwareCollection 替代原来的 Collection
        ...         return TenantAwareCollection(
        ...             name=self.name,
        ...             using="default",  # using 参数会被忽略，实际使用租户连接
        ...             schema=self._SCHEMA,
        ...         )

    注意事项：
    - 传入的 using 参数会被忽略，实际使用的是租户感知的连接别名
    - 第一次访问时会自动注册连接（通过 MilvusClientFactory）
    - 连接别名和配置会缓存在 tenant_info_patch 中，避免重复计算
    """

    def __init__(
        self,
        name: str,
        schema: Optional[CollectionSchema] = None,
        using: str = "default",
        **kwargs,
    ):
        """
        初始化租户感知的 Collection

        Args:
            name: Collection 名称
            schema: Collection schema（可选）
            using: 连接别名（会被忽略，实际使用租户感知的连接）
            **kwargs: 其他参数（传递给父类）

        注意：
            - using 参数会被租户感知的连接别名覆盖
            - 第一次访问时会自动确保租户连接已注册
            - _original_name 保存原始的 name 值，供 property 使用
        """
        # 保存原始的 name（在调用父类 __init__ 之前）
        # 这样如果需要实现租户感知的表名，可以将 _name 改为 property
        self._original_name = name
        self._original_using = using

        # 调用父类构造函数（使用租户感知的 using）
        # 父类会设置 self._name = name
        super().__init__(name=name, schema=schema, using=using, **kwargs)

        logger.debug("创建 TenantAwareCollection [name=%s, using=%s]", name, using)

    def _get_connection(self):
        """
        覆盖父类方法，返回租户感知的连接

        这是核心方法：每次需要访问 Milvus 时都会调用此方法获取连接。
        我们在这里根据租户上下文动态返回正确的连接 handler。

        Returns:
            Milvus 连接 handler

        注意：
            - 此方法会在每次操作时被调用（search、insert、query 等）
            - 我们重新获取租户 using 以支持跨请求的连接切换
        """
        # 动态获取当前租户的连接别名（支持跨请求切换）
        tenant_using = self._get_tenant_aware_using()

        # 确保连接已注册
        self._ensure_connection_registered(tenant_using)

        # 返回对应的连接 handler
        return connections._fetch_handler(tenant_using)

    @staticmethod
    def _get_tenant_aware_using() -> str:
        """
        获取租户感知的连接别名

        根据配置和上下文决定返回哪个连接别名：
        1. 如果启用非租户模式，返回 "default"
        2. 如果启用租户模式，根据当前租户配置返回对应的连接别名
        3. 如果租户模式下没有租户上下文，返回 "default"

        Returns:
            str: pymilvus 连接别名（using）
        """

        def compute_using() -> str:
            """计算租户连接别名"""
            # 从租户配置中获取 Milvus 配置
            milvus_config = get_tenant_milvus_config()
            if not milvus_config:
                raise RuntimeError("租户缺少 Milvus 配置")

            # 基于连接参数生成唯一的连接别名
            cache_key = get_milvus_connection_cache_key(milvus_config)
            return f"tenant_{cache_key}"

        return get_or_compute_tenant_cache(
            patch_key=TenantPatchKey.MILVUS_CONNECTION_CACHE_KEY,
            compute_func=compute_using,
            fallback="default",  # 具体值，不需要延迟计算
            cache_description="Milvus 连接别名",
        )

    @staticmethod
    def _ensure_connection_registered(using: str) -> None:
        """
        确保指定的连接别名已注册

        如果连接尚未注册，会自动注册连接（通过 MilvusClientFactory）。

        Args:
            using: 连接别名

        注意：
            - 对于 "default" 连接，假设已经在应用启动时注册
            - 对于租户连接（tenant_*），如果未注册则自动注册
        """
        # 检查连接是否已存在
        try:
            connections._fetch_handler(using)
            # 连接已存在，直接返回
            return
        except Exception:
            # 连接不存在，需要注册
            pass

        # 如果是默认连接，尝试从环境变量注册
        if using == "default":
            logger.info("📋 注册默认 Milvus 连接")
            config = load_milvus_config_from_env()
            TenantAwareCollection._register_connection(config, using)
            return

        # 租户连接：从租户配置注册
        try:
            tenant_info = get_current_tenant()
            if not tenant_info:
                raise RuntimeError("无法注册租户连接：未设置租户上下文")

            milvus_config = get_tenant_milvus_config()
            if not milvus_config:
                raise RuntimeError(
                    f"无法注册租户连接：租户 {tenant_info.tenant_id} 缺少 Milvus 配置"
                )

            logger.info(
                "📋 为租户 [%s] 注册 Milvus 连接 [using=%s]",
                tenant_info.tenant_id,
                using,
            )

            TenantAwareCollection._register_connection(milvus_config, using)

        except Exception as e:
            logger.error("注册租户连接失败 [using=%s]: %s", using, e)
            raise

    @staticmethod
    def _register_connection(config: dict, using: str) -> None:
        """
        注册 Milvus 连接

        Args:
            config: Milvus 连接配置
            using: 连接别名

        注意：
            - 使用 MilvusClientFactory 来创建连接
            - 这样可以复用现有的连接池管理逻辑
        """
        try:
            # 通过 MilvusClientFactory 创建连接
            # 这样可以复用现有的连接池管理
            factory = get_bean_by_type(MilvusClientFactory)

            # 构建 URI
            host = config.get("host", "localhost")
            port = config.get("port", 19530)
            uri = (
                f"{host}:{port}" if host.startswith("http") else f"http://{host}:{port}"
            )

            # 创建客户端（这会自动注册连接）
            # 注意：不传递 db_name，租户隔离通过 Collection 名称实现
            factory.get_client(
                uri=uri,
                user=config.get("user", ""),
                password=config.get("password", ""),
                alias=using,
            )

            logger.info(
                "✅ Milvus 连接已注册 [using=%s, host=%s, port=%s]", using, host, port
            )

        except Exception as e:
            logger.error("注册 Milvus 连接失败 [using=%s]: %s", using, e)
            raise

    # ============================================================
    # 租户感知的表名支持（可选功能）
    # ============================================================
    # 如果需要支持租户感知的表名，可以取消下面的 @property 注释。
    # 这样不同租户会使用不同的表名，实现表级别的隔离。
    #
    # 注意：启用此功能后，需要确保：
    # 1. 每个租户都有独立的表
    # 2. 表名符合 Milvus 的命名规范
    # 3. 考虑表名长度限制
    #
    @property
    def _name(self) -> str:
        """
        租户感知的表名

        覆盖父类的 _name 属性，为表名添加租户标识。

        示例：
            原始表名: "my_collection"
            租户 A: "tenant_001_my_collection"
            租户 B: "tenant_002_my_collection"

        Returns:
            str: 租户感知的表名
        """
        return self.get_tenant_aware_name(self._original_name)

    @classmethod
    def get_tenant_aware_name(cls, original_name: str) -> str:
        """
        获取租户感知的表名
        """
        return get_tenant_aware_collection_name(original_name)

    @_name.setter
    def _name(self, value: str) -> None:
        """
        设置表名（setter）

        pymilvus 的父类 Collection 可能会尝试设置 _name 属性。
        我们在这里捕获设置操作，更新 _original_name。

        Args:
            value: 要设置的表名
        """
        # 更新原始表名
        # 注意：这里我们存储原始值，而不是租户感知的值
        # 因为 getter 会自动添加租户前缀
        self._original_name = value

    @property
    def using(self) -> str:
        """
        获取租户感知的连接别名
        """
        return self._get_tenant_aware_using()
