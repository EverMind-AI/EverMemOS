"""
MongoDB 生命周期提供者实现
"""

from collections import defaultdict
from fastapi import FastAPI
from typing import Any

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider
from core.oxm.mongo.document_base import DocumentBase
from core.di.utils import get_all_subclasses
from component.mongodb_client_factory import MongoDBClientFactory, MongoDBClientWrapper


logger = get_logger(__name__)


@component(name="mongodb_lifespan_provider")
class MongoDBLifespanProvider(LifespanProvider):
    """MongoDB 生命周期提供者"""

    def __init__(self, name: str = "mongodb", order: int = 15):
        """
        初始化 MongoDB 生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，MongoDB 在数据库连接之后启动
        """
        super().__init__(name, order)
        self._mongodb_factory = None
        self._mongodb_client = None

    async def startup(self, app: FastAPI) -> Any:
        """
        启动 MongoDB 连接和初始化

        Args:
            app (FastAPI): FastAPI应用实例

        Returns:
            Any: MongoDB 客户端信息
        """
        logger.info("正在初始化 MongoDB 连接...")

        try:

            # 获取 MongoDB 客户端工厂
            self._mongodb_factory = get_bean_by_type(MongoDBClientFactory)

            # 获取默认客户端
            self._mongodb_client: MongoDBClientWrapper = (
                await self._mongodb_factory.get_default_client()
            )

            # 手动初始化 Beanie ODM
            all_subclasses_of_document_base = get_all_subclasses(DocumentBase)
            db_document_models = defaultdict(list)
            for subclass in all_subclasses_of_document_base:
                db_document_models[subclass.get_bind_database()].append(subclass)

            # 获取所有的DB名称
            db_names = list(db_document_models.keys())
            db_clients = {
                db_name: await self._mongodb_factory.get_named_client(db_name)
                for db_name in db_names
            }

            # 初始化 Beanie ODM
            for db_name, db_client in db_clients.items():
                await db_client.initialize_beanie(db_document_models[db_name])

            logger.info("✅ MongoDB 连接初始化完成")

        except Exception as e:
            logger.error("❌ MongoDB 初始化过程中出错: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        关闭 MongoDB 连接

        Args:
            app (FastAPI): FastAPI应用实例
        """
        logger.info("正在关闭 MongoDB 连接...")

        if self._mongodb_factory:
            try:
                await self._mongodb_factory.close_all_clients()
                logger.info("✅ MongoDB 连接关闭完成")
            except Exception as e:
                logger.error("❌ 关闭 MongoDB 连接时出错: %s", str(e))
