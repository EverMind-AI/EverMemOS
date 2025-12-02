"""
租户感知的带 Suffix 和 Alias 机制的 Milvus Collection 管理类

本模块结合了 TenantAwareCollection 和 MilvusCollectionWithSuffix 的功能：
1. 租户感知：根据租户上下文自动选择正确的连接和表名
2. 动态表名：支持通过 suffix 或环境变量动态设置表名后缀
3. Alias 机制：真实表名带时间戳，通过 alias 访问
"""

from typing import Optional
from pymilvus import connections, Collection
from pymilvus.client.types import ConsistencyLevel

from core.observation.logger import get_logger
from core.oxm.milvus.milvus_collection_base import (
    MilvusCollectionWithSuffix,
    generate_new_collection_name,
)
from core.tenants.tenantize.oxm.milvus.tenant_aware_collection import (
    TenantAwareCollection,
)
from core.tenants.tenantize.oxm.milvus.config_utils import (
    get_tenant_aware_collection_name,
)
from pymilvus import utility

logger = get_logger(__name__)


class TenantAwareMilvusCollectionWithSuffix(MilvusCollectionWithSuffix):
    """
    租户感知的带 Suffix 和 Alias 机制的 Milvus Collection 管理类

    继承自 MilvusCollectionWithSuffix，增加租户感知能力：
    1. 自动根据租户上下文选择正确的 Milvus 连接
    2. 支持租户感知的表名（自动添加租户前缀）
    3. 保留 MilvusCollectionWithSuffix 的所有功能（suffix、alias、创建管理等）

    核心特性：
    - 租户隔离：不同租户使用不同的连接和表名
    - 动态表名：支持 suffix 和环境变量
    - Alias 机制：真实表名带时间戳，通过 alias 访问
    - 版本管理：可以创建新版本并灰度切换
    - 租户前缀：所有操作都会自动添加租户前缀（如 tenant_001_movies）

    表名规则：
    - 原始基础名: movies
    - 带 suffix: movies_production
    - 带租户前缀: tenant_001_movies_production  (alias)
    - 真实表名: tenant_001_movies_production-20231015123456789000

    使用方式：
    1. 子类定义：
       - _COLLECTION_NAME: Collection 的基础名称（必需）
       - _SCHEMA: Collection 的 Schema 定义（必需）
       - _INDEX_CONFIGS: 索引配置列表（可选）
       - _DB_USING: Milvus 连接别名（可选，会被租户感知的连接覆盖）

    2. 实例化：
       mgr = TenantAwareMovieCollection(suffix="customer_a")
       # 在租户上下文中：
       # - 使用租户的 Milvus 连接
       # - Alias: tenant_001_movies_customer_a
       # - 真实名: tenant_001_movies_customer_a-20231015123456789000

    3. 初始化：
       with tenant_context(tenant_info):
           mgr.ensure_all()  # 一键初始化

    4. 使用：
       with tenant_context(tenant_info):
           mgr.collection.insert([...])
           mgr.collection.search(...)

    示例：
        class MovieCollection(TenantAwareMilvusCollectionWithSuffix):
            _COLLECTION_NAME = "movies"
            _SCHEMA = CollectionSchema(fields=[...])
            _INDEX_CONFIGS = [
                IndexConfig(field_name="embedding", index_type="IVF_FLAT", ...),
            ]

        # 多租户场景使用
        tenant_a = TenantInfo(tenant_id="tenant_001", ...)
        tenant_b = TenantInfo(tenant_id="tenant_002", ...)

        mgr = MovieCollection(suffix="production")

        # 租户 A 的操作
        with tenant_context(tenant_a):
            mgr.ensure_all()
            mgr.collection.insert([...])

        # 租户 B 的操作
        with tenant_context(tenant_b):
            mgr.ensure_all()
            mgr.collection.insert([...])
    """

    def __init__(self, suffix: Optional[str] = None):
        """
        初始化租户感知的 Collection 管理器

        Args:
            suffix: Collection 名称后缀，如果不提供则从环境变量读取

        注意：
            - 保存原始的 _alias_name（不带租户前缀）
            - 实际使用的表名会在运行时动态添加租户前缀
        """
        super().__init__(suffix=suffix)
        # 保存原始的 alias 名称（不带租户前缀）
        # 用于在 name property 中动态计算租户感知的名称
        self._original_alias_name = self._alias_name

    @property
    def name(self) -> str:
        """
        获取租户感知的 Collection 名称（alias）

        覆盖父类的 name property，动态添加租户前缀。
        这样所有使用 self.name 的地方都会自动获得租户感知的表名。

        Returns:
            str: 租户感知的 alias 名称

        示例：
            原始 alias: movies_production
            租户 A: tenant_001_movies_production
            租户 B: tenant_002_movies_production
        """
        return TenantAwareCollection.get_tenant_aware_name(self._original_alias_name)

    @property
    def using(self) -> str:
        """
        获取租户感知的连接别名
        """
        return TenantAwareCollection._get_tenant_aware_using()

    def ensure_connection_registered(self) -> None:
        """
        确保租户感知的连接已注册
        """
        TenantAwareCollection._ensure_connection_registered(self.using)

    def load_collection(self) -> TenantAwareCollection:
        """
        加载或创建租户感知的 Collection

        覆盖父类方法，使用 TenantAwareCollection 替代普通的 Collection。
        这样可以确保所有 Collection 操作都是租户感知的。

        Args:
            name: Collection 名称（alias 名称，已包含租户前缀）

        Returns:
            TenantAwareCollection 实例

        注意：
            - 使用 TenantAwareCollection 自动处理租户连接
            - 保持 MilvusCollectionWithSuffix 的 alias 机制
            - 如果 alias 不存在，会创建新的带时间戳的 Collection
            - name 参数应该已经是租户感知的（通过 self.name 传入）
        """
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        new_real_name = generate_new_collection_name(origin_alias_name)
        tenant_aware_new_real_name = get_tenant_aware_collection_name(new_real_name)

        # 先探测 alias 是否存在（使用租户感知的连接）
        # 注意：TenantAwareCollection 会自动处理 using 参数
        self.ensure_connection_registered()

        if not utility.has_collection(tenant_aware_alias_name, using=using):
            # Collection 不存在，创建新的带时间戳的 Collection
            logger.info(
                "Collection '%s' 不存在，创建新的租户感知 Collection: %s",
                origin_alias_name,
                tenant_aware_new_real_name,
            )

            # 创建租户感知的 Collection
            # 使用原生的 Collection，需要显式传入 using 参数
            _coll = Collection(
                name=tenant_aware_new_real_name,
                schema=self._SCHEMA,
                consistency_level=ConsistencyLevel.Bounded,
                using=using,
            )

            # 创建 alias 指向新 Collection
            # 注意：先删除可能存在的旧 alias
            try:
                utility.drop_alias(tenant_aware_alias_name, using=using)
            except Exception:
                pass  # alias 不存在，忽略

            utility.create_alias(
                collection_name=tenant_aware_new_real_name,
                alias=tenant_aware_alias_name,
                using=using,
            )
            logger.info(
                "已创建 Alias '%s' -> '%s'",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )

        # 统一通过 alias 加载租户感知的 Collection
        coll = TenantAwareCollection(
            name=origin_alias_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )

        return coll

    def ensure_create(self) -> None:
        """
        确保 Collection 已创建

        覆盖父类方法，使用租户感知的 alias 名称。

        此方法会触发 Collection 的懒加载，如果 alias 不存在则创建新 Collection。
        """
        if self._collection_instance is None:
            # 使用租户感知的 alias 名称
            self._collection_instance = self.load_collection()
        logger.info("Collection '%s' 已就绪", self.name)

    def create_new_collection(self) -> TenantAwareCollection:
        """
        创建一个新的租户感知的真实 Collection（不切换 alias）

        覆盖父类方法，使用 TenantAwareCollection 和租户感知的名称。

        Returns:
            新的租户感知 Collection 实例（已创建索引并 load）

        注意：
            - 使用原生 Collection 创建（需要显式传入 using 参数）
            - 新 Collection 的名称包含租户前缀和时间戳
            - 返回 TenantAwareCollection 实例确保租户隔离
            - 自动创建索引并加载到内存
        """
        if not self._SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__} 必须定义 '_SCHEMA' 以支持集合创建"
            )

        # 使用租户感知的 alias 名称
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        new_real_name = generate_new_collection_name(origin_alias_name)
        tenant_aware_new_real_name = get_tenant_aware_collection_name(new_real_name)

        # 创建新的租户感知集合
        # 使用原生的 Collection，需要显式传入 using 参数
        _coll = Collection(
            name=tenant_aware_new_real_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
            using=using,
        )

        logger.info("已创建新的租户感知 Collection: %s", tenant_aware_new_real_name)

        # 为新集合创建索引并加载
        try:
            self._create_indexes_for_collection(_coll)
            _coll.load()
            logger.info("为新 Collection '%s' 创建索引并加载完成", new_real_name)
        except Exception as e:
            logger.warning("为新集合创建索引时出错: %s", e)
            raise

        # 返回 TenantAwareCollection 实例，使用原始 alias 名称
        # 注意：这里使用 _original_alias_name，TenantAwareCollection 会自动添加租户前缀
        new_coll = TenantAwareCollection(
            name=new_real_name,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )

        return new_coll

    def switch_alias(
        self, new_collection: TenantAwareCollection, drop_old: bool = False
    ) -> None:
        """
        将 alias 切换到指定的新集合，并可选删除旧集合

        覆盖父类方法，使用租户感知的 alias 名称。

        Args:
            new_collection: 新的 Collection 实例
            drop_old: 是否删除旧集合（默认 False）

        注意：
            - 使用租户感知的 alias 名称进行切换
            - 优先使用 alter_alias，失败则使用 drop/create
            - 切换后刷新类级缓存
        """
        # 使用租户感知的 alias 名称
        using = self.using
        origin_alias_name = self._original_alias_name
        tenant_aware_alias_name = get_tenant_aware_collection_name(origin_alias_name)
        tenant_aware_new_real_name = new_collection.name

        # 获取旧集合真实名（若存在）
        old_real_name: Optional[str] = None
        try:
            conn = connections._fetch_handler(using)
            desc = conn.describe_alias(tenant_aware_alias_name)
            old_real_name = (
                desc.get("collection_name") if isinstance(desc, dict) else None
            )
        except Exception:
            old_real_name = None

        # 别名切换
        try:
            conn = connections._fetch_handler(using)
            conn.alter_alias(tenant_aware_new_real_name, tenant_aware_alias_name)
            logger.info(
                "已将别名 '%s' 切换至 '%s'",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )
        except Exception as e:
            logger.warning("alter_alias 失败，尝试 drop/create: %s", e)
            try:
                utility.drop_alias(tenant_aware_alias_name, using=using)
            except Exception:
                pass
            utility.create_alias(
                collection_name=tenant_aware_new_real_name,
                alias=tenant_aware_alias_name,
                using=using,
            )
            logger.info(
                "已创建别名 '%s' -> '%s'",
                tenant_aware_alias_name,
                tenant_aware_new_real_name,
            )

        # 可选删除旧集合（在切换完成之后执行）
        if drop_old and old_real_name:
            try:
                utility.drop_collection(old_real_name, using=using)
                logger.info("已删除旧集合: %s", old_real_name)
            except Exception as e:
                logger.warning("删除旧集合失败（可手动处理）: %s", e)

        # 刷新类级别缓存为 alias 集合
        try:
            self.__class__._collection_instance = TenantAwareCollection(
                name=origin_alias_name,
                schema=self._SCHEMA,
                consistency_level=ConsistencyLevel.Bounded,
            )
        except Exception:
            pass

    def exists(self) -> bool:
        """
        检查 Collection 是否存在（通过 alias）

        覆盖父类方法，使用租户感知的 name 和 using。

        Returns:
            bool: Collection 是否存在
        """
        name = self.name
        using = self.using
        return utility.has_collection(name, using=using)

    def drop(self) -> None:
        """
        删除当前 Collection（包括 alias 和真实 Collection）

        覆盖父类方法，使用租户感知的 name、using 和 TenantAwareCollection。

        注意：
            - 使用租户感知的连接别名
            - 使用 TenantAwareCollection 确保租户隔离
            - 删除的是真实的 Collection（不是 alias）
        """
        using = self.using
        name = self.name
        try:
            utility.drop_collection(name, using=using)
            logger.info("已删除 Collection '%s'", name)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Collection '%s' 不存在或删除失败: %s", name, e)
