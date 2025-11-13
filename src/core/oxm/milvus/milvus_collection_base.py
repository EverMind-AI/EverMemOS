import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pymilvus import Collection, DataType, FieldSchema, utility, CollectionSchema
from pymilvus.client.types import ConsistencyLevel, LoadState

from pymilvus import connections
from core.oxm.milvus.async_collection import AsyncCollection
from common_utils.datetime_utils import get_now_with_timezone

logger = logging.getLogger(__name__)


def generate_new_collection_name(alias: str) -> str:
    """基于别名生成带时间戳的新集合名称。"""
    now = get_now_with_timezone()
    return f"{alias}_{now.strftime('%Y%m%d%H%M%S%f')}"


@dataclass
class IndexConfig:
    """
    索引配置类

    用于定义需要创建的索引（支持向量索引和标量索引）

    Attributes:
        field_name: 字段名
        index_type: 索引类型（如 IVF_FLAT, HNSW, AUTOINDEX 等）
        metric_type: 度量类型（向量索引必需，如 L2, COSINE, IP）
        params: 索引参数（可选）
        index_name: 索引名称（可选，不指定则自动生成）

    示例：
        # 向量索引
        IndexConfig(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128}
        )

        # 标量索引
        IndexConfig(
            field_name="title",
            index_type="AUTOINDEX"
        )
    """

    field_name: str
    index_type: str
    metric_type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    index_name: Optional[str] = None

    def to_index_params(self) -> Dict[str, Any]:
        """转换为 pymilvus 的索引参数格式"""
        result = {"index_type": self.index_type}
        if self.metric_type:
            result["metric_type"] = self.metric_type
        if self.params:
            result["params"] = self.params
        return result


def get_collection_suffix(suffix: Optional[str] = None) -> str:
    """
    获取 Collection 名称后缀，用于多租户场景

    Args:
        suffix: 显式传入的后缀，如果提供则直接返回
               如果不提供，从环境变量 SELF_MILVUS_COLLECTION_NS 读取

    Returns:
        Collection 后缀字符串，如果都未设置则返回空字符串
    """
    if suffix is not None:
        return suffix
    return os.getenv("SELF_MILVUS_COLLECTION_NS", "")


class MilvusCollectionBase:
    """
    Milvus Collection 基础管理类

    职责：
    1. 管理 Collection 的基本信息（名称、Schema、索引配置）
    2. 提供懒加载的 Collection 实例（内部缓存）
    3. 提供工具方法（ensure_indexes、ensure_loaded）

    适用场景：
    - 简单的 Collection 管理
    - 只读数据源（由其他团队管理，只需要查询）
    - 不需要 suffix、时间戳、alias 等复杂逻辑

    使用方式：
    1. 子类定义：
       - _COLLECTION_NAME: Collection 名称（必需）
       - _SCHEMA: Collection Schema（可选）
       - _INDEX_CONFIGS: 索引配置列表（可选）
       - _DB_USING: Milvus 连接别名（可选，默认 "default"）

    2. 实例化：
       mgr = MovieCollection()  # 使用类定义的 _DB_USING
       # 或
       mgr = MovieCollection(using="custom_db")  # 覆盖类定义

    3. 使用：
       mgr.ensure_loaded()  # 加载到内存
       mgr.collection.search(...)  # 使用 Collection

    示例：
        # 只读场景（数据源由其他团队管理）
        class ReadOnlyMovieCollection(MilvusCollectionBase):
            _COLLECTION_NAME = "external_movies"  # 固定的 Collection 名称
            _DB_USING = "external_db"  # 使用外部数据库连接

        mgr = ReadOnlyMovieCollection()
        mgr.ensure_loaded()
        results = mgr.collection.search(...)
    """

    # 子类必须定义的属性
    _COLLECTION_NAME: Optional[str] = None

    # 子类可选定义的属性
    _SCHEMA: Optional[CollectionSchema] = None
    _INDEX_CONFIGS: Optional[List[IndexConfig]] = None
    _DB_USING: Optional[str] = "default"

    # 类级别的实例缓存
    _collection_name: Optional[str] = None
    _collection_instance: Optional[Collection] = None
    _async_collection_instance: Optional[AsyncCollection] = None

    def __init__(self):
        """初始化配置容器"""
        if not self._COLLECTION_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__} 必须定义 '_COLLECTION_NAME' 类属性"
            )

        # 使用类属性 _DB_USING，如果未定义则使用 "default"
        self._using = self._DB_USING if self._DB_USING is not None else "default"
        # 缓存 collection 描述，避免属性在 __init__ 外首次定义
        self._collection_desc: Optional[Dict[str, Any]] = None

    @classmethod
    def collection(cls) -> Collection:
        """获取 Collection 实例（类级别缓存）"""
        if cls._collection_instance is None:
            raise ValueError(
                f"{cls.__name__} 的 Collection 实例未创建，请先调用 ensure_loaded()"
            )
        return cls._collection_instance

    @classmethod
    def async_collection(cls) -> AsyncCollection:
        """获取异步 Collection 实例（类级别缓存）"""
        if cls._async_collection_instance is None:
            if cls._collection_instance is None:
                raise ValueError(
                    f"{cls.__name__} 的 Collection 实例未创建，请先调用 ensure_loaded()"
                )
            cls._async_collection_instance = AsyncCollection(cls._collection_instance)
        return cls._async_collection_instance

    @property
    def name(self) -> str:
        """获取实际的 Collection 名称"""
        return self._COLLECTION_NAME

    @property
    def real_name(self) -> str:
        if self._collection_name:
            return self._collection_name
        else:
            raise ValueError("Collection 名称未获取，请先调用 ensure_collection_desc()")

    @property
    def using(self) -> str:
        """获取连接别名"""
        return self._DB_USING if self._DB_USING is not None else "default"

    def load_collection(self, name: str) -> Collection:
        """加载 Collection（内部方法）"""
        if not utility.has_collection(name, using=self.using):
            raise ValueError(f"Collection '{name}' 不存在")

        coll = Collection(
            name=name,
            using=self.using,
            schema=self._SCHEMA,
            consistency_level=ConsistencyLevel.Bounded,
        )
        logger.info("加载 Collection '%s'", name)
        return coll

    def ensure_loaded(self) -> None:
        """确保 Collection 已加载到内存（类级别缓存）"""
        # 懒加载 Collection
        if self.__class__._collection_instance is None:
            self.__class__._collection_instance = self.load_collection(self.name)

        coll = self.__class__._collection_instance

        try:
            load_state = utility.load_state(coll.name, using=self.using)

            if load_state == LoadState.NotLoad:
                logger.info("Collection '%s' 未加载，正在加载到内存...", self.name)
                coll.load()
                logger.info("Collection '%s' 加载成功", self.name)
            elif load_state == LoadState.Loading:
                logger.info("Collection '%s' 正在加载中，等待加载完成...", self.name)
                coll.load()
            else:
                logger.info("Collection '%s' 已加载", self.name)

        except Exception as e:
            logger.error("加载 Collection 时出错: %s", e)
            raise

    def ensure_indexes(self) -> None:
        """创建所有配置的索引（diff 方式）"""
        if not self._INDEX_CONFIGS:
            logger.info("Collection '%s' 未配置索引，跳过", self.name)
            return

        # 懒加载 Collection
        if self._collection_instance is None:
            self._collection_instance = self.load_collection(self.name)

        coll = self._collection_instance
        self._create_indexes_for_collection(coll)

    @staticmethod
    def _get_existing_indexes(coll: Collection) -> Dict[str, Dict[str, Any]]:
        """获取指定 Collection 中已存在的索引信息"""
        try:
            indexes_info = coll.indexes
            result = {}

            for index in indexes_info:
                field_name = index.field_name
                result[field_name] = {
                    "index_type": index.params.get("index_type"),
                    "metric_type": index.params.get("metric_type"),
                }

            return result

        except Exception as e:  # pylint: disable=broad-except
            logger.warning("获取索引信息时出错: %s", e)
            return {}

    def _create_indexes_for_collection(self, coll: Collection) -> None:
        """为指定 Collection 创建缺失的索引（复用 ensure_indexes 的 diff 逻辑）"""
        try:
            existing_indexes = self._get_existing_indexes(coll)
            existing_field_names = set(existing_indexes.keys())

            logger.info(
                "Collection '%s' 已存在的索引字段: %s", coll.name, existing_field_names
            )

            index_configs = self._INDEX_CONFIGS or []
            for index_config in index_configs:
                field_name = index_config.field_name
                if field_name in existing_field_names:
                    logger.info("字段 '%s' 已有索引，跳过", field_name)
                    continue

                logger.info(
                    "为字段 '%s' 创建索引（类型: %s）...",
                    field_name,
                    index_config.index_type,
                )
                create_kwargs = {
                    "field_name": field_name,
                    "index_params": index_config.to_index_params(),
                    "timeout": 120,
                }
                if index_config.index_name:
                    create_kwargs["index_name"] = index_config.index_name
                coll.create_index(**create_kwargs)
                logger.info("字段 '%s' 索引创建成功", field_name)

            logger.info("Collection '%s' 索引检查与创建完成", coll.name)
        except Exception as e:
            logger.error("为 Collection '%s' 创建索引时出错: %s", coll.name, e)
            raise

    @staticmethod
    def _get_collection_desc(collection_: Collection) -> Dict[str, Any]:
        conn = collection_._get_connection()
        return conn.describe_collection(collection_.name)

    def ensure_collection_desc(self) -> Dict[str, Any]:
        collection_desc = self._get_collection_desc(self._collection_instance)
        self._collection_desc = collection_desc
        self._collection_name = collection_desc.get("collection_name")

    def ensure_all(self) -> None:
        """
        一键完成所有初始化操作

        执行顺序：
        1. ensure_loaded(): 加载到内存
        2. ensure_collection_desc(): 获取 Collection 描述
        """
        self.ensure_loaded()
        self.ensure_collection_desc()
        logger.info("Collection '%s' 初始化完成，真实名: %s", self.name, self.real_name)


class MilvusCollectionWithSuffix(MilvusCollectionBase):
    """
    带 Suffix 和 Alias 机制的 Milvus Collection 管理类

    继承自 MilvusCollectionBase，增加：
    1. 动态表名：支持通过 suffix 或环境变量动态设置表名后缀（多租户场景）
    2. Alias 机制：真实表名带时间戳，通过 alias 访问（方便后续切换）
       - Alias: {base_name}_{suffix}
       - 真实名: {base_name}_{suffix}-{timestamp}
    3. 创建管理：提供 ensure_create、ensure_all 等方法

    适用场景：
    - 多租户场景，需要为不同客户创建独立的 Collection
    - 需要版本管理，保留历史 Collection
    - 需要灰度切换，通过 alias 切换不同版本

    使用方式：
    1. 子类定义：
       - _BASE_NAME: Collection 的基础名称（必需）
       - _SCHEMA: Collection 的 Schema 定义（必需）
       - _INDEX_CONFIGS: 索引配置列表（可选）
       - _DB_USING: Milvus 连接别名（可选，默认 "default"）

    2. 实例化：
       mgr = MovieCollection(suffix="customer_a")
       # Alias: movies_customer_a
       # 真实名: movies_customer_a-20231015123456789000

       # 或指定数据库连接
       mgr = MovieCollection(suffix="customer_a", using="custom_db")

    3. 初始化：
       mgr.ensure_all()  # 一键初始化

    4. 使用：
       mgr.collection.insert([...])
       mgr.collection.search(...)

    示例：
        class MovieCollection(MilvusCollectionWithSuffix):
            _BASE_NAME = "movies"
            _SCHEMA = CollectionSchema(fields=[...])
            _INDEX_CONFIGS = [
                IndexConfig(field_name="embedding", index_type="IVF_FLAT", ...),
                IndexConfig(field_name="year", index_type="AUTOINDEX")
            ]
            _DB_USING = "my_milvus"  # 可选：指定默认数据库连接

        # 使用
        mgr = MovieCollection(suffix="customer_a")
        mgr.ensure_all()
        mgr.collection.insert([...])
    """

    def __init__(self, suffix: Optional[str] = None):
        """
        初始化配置容器

        Args:
            suffix: Collection 名称后缀，如果不提供则从环境变量 SELF_MILVUS_COLLECTION_NS 读取
        """
        if not self._COLLECTION_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__} 必须定义 '_COLLECTION_NAME' 类属性"
            )

        if not self._SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__} 必须定义 '_SCHEMA' 类属性（创建场景必需）"
            )

        # 获取 suffix（支持传参或从环境变量获取）
        self._suffix = get_collection_suffix(suffix)

        # 构造 alias 名称
        if self._suffix:
            self._alias_name = f"{self._COLLECTION_NAME}_{self._suffix}"
        else:
            self._alias_name = self._COLLECTION_NAME

        # 调用基类初始化
        super().__init__()

    @property
    def name(self) -> str:
        """获取 Collection 名称"""
        return self._alias_name

    def load_collection(self, name: str) -> Collection:
        """
        加载或创建 Collection（内部方法）

        覆盖基类方法，增加创建逻辑
        """
        # 先探测 alias 是否存在

        if not utility.has_collection(name, using=self._using):
            # Collection 不存在，创建新的带时间戳的 Collection
            _collection_name = generate_new_collection_name(name)

            logger.info(
                "Collection '%s' 不存在，创建新 Collection: %s", name, _collection_name
            )

            # 创建 Collection
            Collection(
                name=_collection_name,
                schema=self._SCHEMA,
                using=self._using,
                consistency_level=ConsistencyLevel.Bounded,  # 默认 bounded 一致性
            )

            # 创建 alias 指向新 Collection
            # 删除指向的实际 Collection的时候，alias并不会自动删除，需要先删除alias
            utility.drop_alias(name, using=self._using)
            utility.create_alias(
                collection_name=_collection_name, alias=name, using=self._using
            )
            logger.info("已创建 Alias '%s' -> '%s'", name, _collection_name)

        # 统一通过 alias 加载（无论已存在还是新创建）
        coll = Collection(name=name, using=self._using)

        return coll

    def ensure_create(self) -> None:
        """
        确保 Collection 已创建

        此方法会触发 Collection 的懒加载，如果 alias 不存在则创建新 Collection
        """
        if self._collection_instance is None:
            self._collection_instance = self.load_collection(self._alias_name)
        logger.info("Collection '%s' 已就绪", self.name)

    def ensure_all(self) -> None:
        """
        一键完成所有初始化操作

        执行顺序：
        1. ensure_create(): 创建 Collection 和 alias
        2. ensure_indexes(): 创建所有配置的索引
        3. ensure_loaded(): 加载到内存
        """
        logger.info("开始初始化 Collection '%s'", self.name)

        self.ensure_create()
        self.ensure_collection_desc()
        self.ensure_indexes()
        self.ensure_loaded()

        logger.info("Collection '%s' 初始化完成，真实名: %s", self.name, self.real_name)

    def create_new_collection(self) -> Collection:
        """
        创建一个新的真实 Collection（不切换 alias）。
        - 使用类定义的 `_SCHEMA` 创建新集合
        - 按 `_INDEX_CONFIGS` 为新集合创建索引并 load

        Returns:
            新集合实例（已创建索引并 load）
        """
        if not self._SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__} 必须定义 '_SCHEMA' 以支持集合创建"
            )

        alias_name = self._alias_name

        # 创建新集合
        new_real_name = generate_new_collection_name(alias_name)
        Collection(
            name=new_real_name,
            schema=self._SCHEMA,
            using=self._using,
            consistency_level=ConsistencyLevel.Bounded,
        )

        # 为新集合创建索引
        try:
            new_coll = Collection(name=new_real_name, using=self._using)
            self._create_indexes_for_collection(new_coll)
            new_coll.load()
        except Exception as e:
            logger.warning("为新集合创建索引时出错，可忽略: %s", e)

        return new_coll

    def switch_alias(self, new_collection: Collection, drop_old: bool = False) -> None:
        """
        将 alias 切换到指定的新集合，并可选删除旧集合。
        - 优先 alter_alias，失败则走 drop/create
        - 切换后刷新类级缓存
        """
        alias_name = self._alias_name
        new_real_name = new_collection.name

        # 获取旧集合真实名（若存在）
        old_real_name: Optional[str] = None
        try:
            conn = connections._fetch_handler(self._using)
            desc = conn.describe_alias(alias_name)
            old_real_name = (
                desc.get("collection_name") if isinstance(desc, dict) else None
            )
        except Exception:
            old_real_name = None

        # 别名切换
        try:
            conn = connections._fetch_handler(self._using)
            conn.alter_alias(new_real_name, alias_name)
            logger.info("已将别名 '%s' 切换至 '%s'", alias_name, new_real_name)
        except Exception as e:
            logger.warning("alter_alias 失败，尝试 drop/create: %s", e)
            try:
                utility.drop_alias(alias_name, using=self._using)
            except Exception:
                pass
            utility.create_alias(
                collection_name=new_real_name, alias=alias_name, using=self._using
            )
            logger.info("已创建别名 '%s' -> '%s'", alias_name, new_real_name)

        # 可选删除旧集合（在切换完成之后执行）
        if drop_old and old_real_name:
            try:
                utility.drop_collection(old_real_name, using=self._using)
                logger.info("已删除旧集合: %s", old_real_name)
            except Exception as e:
                logger.warning("删除旧集合失败（可手动处理）: %s", e)

        # 刷新类级别缓存为 alias 集合
        try:
            self.__class__._collection_instance = Collection(
                name=alias_name, using=self._using
            )
            self.ensure_collection_desc()
        except Exception:
            pass

    def exists(self) -> bool:
        """检查 Collection 是否存在（通过 alias）"""
        return utility.has_collection(self.name, using=self._using)

    def drop(self) -> None:
        """删除当前 Collection（包括 alias 和真实 Collection）"""
        try:
            if not self._collection_instance:
                self._collection_instance = Collection(
                    name=self.name, using=self._using
                )

            real_name = self._collection_instance.name
            logger.info("找到 Collection '%s' 对应的真实名称: %s", self.name, real_name)

            utility.drop_collection(real_name, using=self._using)
            logger.info("已删除 Collection '%s'", real_name)

        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Collection '%s' 不存在或删除失败: %s", self.name, e)


if __name__ == "__main__":
    connections.connect("default", host="10.241.132.75", port=19530)

    class TestCollection(MilvusCollectionWithSuffix):
        _COLLECTION_NAME = "test"
        _SCHEMA = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="updated_at", dtype=DataType.INT64),
                FieldSchema(name="deleted_at", dtype=DataType.INT64),
                FieldSchema(name="deleted_by", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(
                    name="deleted_reason", dtype=DataType.VARCHAR, max_length=255
                ),
            ]
        )
        _INDEX_CONFIGS = [
            IndexConfig(
                field_name="vector",
                index_type="HNSW",  # 高效的近似最近邻搜索
                metric_type="COSINE",  # 欧氏距离
            params={
                "M": 16,  # 每个节点的最大边数
                "efConstruction": 200,  # 构建时的搜索宽度
            })
        ]
        _DB_USING = "default"

    collection = TestCollection(suffix="zhanghui")
    collection.ensure_all()
    assert collection.name == "test_zhanghui"
    assert collection.real_name != "test_zhanghui"

    class TestCollection2(MilvusCollectionBase):
        _COLLECTION_NAME = "test_zhanghui"
        _SCHEMA = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=255),
            ]
        )
        _INDEX_CONFIGS = [
            IndexConfig(
                field_name="vector",
                index_type="HNSW",  # 高效的近似最近邻搜索
                metric_type="COSINE",  # 欧氏距离
                params={
                  "M": 16,  # 每个节点的最大边数
                  "efConstruction": 200,  # 构建时的搜索宽度
                })
        ]
        _DB_USING = "default"

    collection2 = TestCollection2()
    collection2.ensure_all()
    assert collection2.name == "test_zhanghui"
    assert collection2.real_name != "test_zhanghui"

    assert collection2.real_name == collection.real_name

    import ipdb

    ipdb.set_trace()
    import asyncio

    asyncio.run(TestCollection.async_collection().insert([[1, 2, 3], [4, 5, 6]]))
