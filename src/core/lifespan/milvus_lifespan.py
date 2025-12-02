"""
Milvus 生命周期提供者实现
"""

from collections import defaultdict
from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger
from core.di.utils import get_bean, get_all_subclasses
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.oxm.milvus.milvus_collection_base import MilvusCollectionBase

logger = get_logger(__name__)


@component(name="milvus_lifespan_provider")
class MilvusLifespanProvider(LifespanProvider):
    """Milvus 生命周期提供者"""

    def __init__(self, name: str = "milvus", order: int = 20):
        """
        初始化 Milvus 生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，Milvus 在数据库连接之后启动
        """
        super().__init__(name, order)
        self._milvus_factory = None
        self._milvus_clients = {}

    async def startup(self, app: FastAPI) -> Any:
        """
        启动 Milvus 连接和初始化

        Args:
            app (FastAPI): FastAPI应用实例

        Returns:
            Any: Milvus 客户端信息
        """
        logger.info("正在初始化 Milvus 连接...")

        try:
            # 获取 Milvus 客户端工厂
            self._milvus_factory = get_bean("milvus_client_factory")

            # 获取所有具体的 Collection 类（通过检查 _COLLECTION_NAME 是否存在）
            all_collection_classes = [
                cls
                for cls in get_all_subclasses(MilvusCollectionBase)
                if cls._COLLECTION_NAME is not None  # 有 _COLLECTION_NAME 的是具体类
            ]

            # 按 using 分组
            using_collections = defaultdict(list)
            for collection_class in all_collection_classes:
                using = collection_class._DB_USING
                using_collections[using].append(collection_class)
                logger.info(
                    "发现 Collection 类: %s [using=%s]",
                    collection_class.__name__,
                    using,
                )

            # 获取所有需要的客户端
            for using, collection_classes in using_collections.items():
                # 获取客户端
                client = self._milvus_factory.get_named_client(using)
                self._milvus_clients[using] = client

                # 初始化每个 Collection
                for collection_class in collection_classes:
                    try:
                        collection = collection_class()
                        collection.ensure_all()
                        logger.info(
                            "✅ Collection '%s' 初始化成功 [using=%s]",
                            collection.name,
                            using,
                        )
                    except Exception as e:
                        logger.error(
                            "❌ Collection '%s' 初始化失败 [using=%s]: %s",
                            collection_class._COLLECTION_NAME,
                            using,
                            e,
                        )
                        raise
            logger.info("✅ Milvus 连接初始化完成")

        except Exception as e:
            logger.error("❌ Milvus 初始化过程中出错: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        关闭 Milvus 连接

        Args:
            app (FastAPI): FastAPI应用实例
        """
        logger.info("正在关闭 Milvus 连接...")

        if self._milvus_factory:
            try:
                self._milvus_factory.close_all_clients()
                logger.info("✅ Milvus 连接关闭完成")
            except Exception as e:
                logger.error("❌ 关闭 Milvus 连接时出错: %s", str(e))

        # 清理 app.state 中的 Milvus 相关属性
        for attr in ['milvus_clients', 'milvus_factory']:
            if hasattr(app.state, attr):
                delattr(app.state, attr)
