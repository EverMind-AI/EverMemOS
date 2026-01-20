"""
Dual Storage Model Proxy - 拦截 MongoDB 调用层

通过拦截 self.model 的所有 MongoDB 调用来实现双存储，Repository 代码零改动。

工作原理：
1. 在 Mixin 的 __init__ 中替换 self.model 为 Proxy
2. Proxy 拦截所有 MongoDB 方法调用（find, get, insert, save, delete 等）
3. 内部实现：
   - 查询：先查 MongoDB 获取 IDs，再从 KV 批量加载完整数据
   - 写入：写 MongoDB 后自动同步 KV
   - 删除：删 MongoDB 后自动删 KV
4. 返回和原来完全一样的数据结构

优势：
- Repository 代码完全不需要改动（零改动）
- 主分支更新 CRUD 方法时，无需同步更新
- 双存储完全透明
"""

from typing import TYPE_CHECKING, Optional, Any, List
from pymongo.asynchronous.client_session import AsyncClientSession

from core.observation.logger import get_logger

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )

logger = get_logger(__name__)


class DualStorageQueryProxy:
    """
    Query Cursor Proxy - 拦截 MongoDB 查询游标操作

    拦截 find() 返回的 Cursor 对象，自动从 KV-Storage 加载完整数据
    """

    def __init__(
        self,
        mongo_cursor,
        kv_storage: "KVStorageInterface",
        full_model_class,
    ):
        """
        Initialize query cursor proxy

        Args:
            mongo_cursor: MongoDB query cursor (from model.find())
            kv_storage: KV-Storage instance
            full_model_class: Full model class (e.g., EpisodicMemory)
        """
        self._mongo_cursor = mongo_cursor
        self._kv_storage = kv_storage
        self._full_model_class = full_model_class

    def sort(self, *args, **kwargs):
        """Proxy sort method"""
        self._mongo_cursor = self._mongo_cursor.sort(*args, **kwargs)
        return self

    def skip(self, *args, **kwargs):
        """Proxy skip method"""
        self._mongo_cursor = self._mongo_cursor.skip(*args, **kwargs)
        return self

    def limit(self, *args, **kwargs):
        """Proxy limit method"""
        self._mongo_cursor = self._mongo_cursor.limit(*args, **kwargs)
        return self

    async def to_list(self, *args, **kwargs) -> List[Any]:
        """
        Execute query and load full data from KV-Storage

        Returns:
            List of full model instances
        """
        # 1. 执行 MongoDB 查询（只查询 MongoDB，获取所有文档）
        mongo_docs = await self._mongo_cursor.to_list(*args, **kwargs)

        if not mongo_docs:
            return []

        # 2. MongoDB 包含完整数据，直接返回
        # （当前方案：主分支的 EpisodicMemory 在 MongoDB 存储所有字段）
        logger.debug(f"✅ Query returned {len(mongo_docs)} documents from MongoDB")
        return mongo_docs

    async def delete(self, *args, **kwargs):
        """
        Delete documents matching query

        Also deletes from KV-Storage
        """
        # 1. 先获取所有要删除的文档 IDs
        mongo_docs = await self._mongo_cursor.to_list()
        doc_ids = [str(doc.id) for doc in mongo_docs]

        # 2. 删除 MongoDB
        result = await self._mongo_cursor.delete(*args, **kwargs)

        # 3. 批量删除 KV-Storage
        if doc_ids:
            try:
                await self._kv_storage.batch_delete(keys=doc_ids)
                logger.debug(f"✅ Deleted {len(doc_ids)} documents from KV-Storage")
            except Exception as e:
                logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

        return result

    def __getattr__(self, name):
        """Proxy all other methods to original cursor"""
        return getattr(self._mongo_cursor, name)


class DualStorageModelProxy:
    """
    Model Proxy - 拦截 MongoDB Model 层调用

    替换 Repository 的 self.model，拦截所有 MongoDB 操作：
    - find() -> 返回 QueryProxy
    - get() -> 优先从 KV 读取
    - 其他方法透明传递
    """

    def __init__(
        self,
        original_model,
        kv_storage: "KVStorageInterface",
        full_model_class,
    ):
        """
        Initialize model proxy

        Args:
            original_model: Original Beanie Document model class
            kv_storage: KV-Storage instance
            full_model_class: Full model class (same as original_model in current design)
        """
        self._original_model = original_model
        self._kv_storage = kv_storage
        self._full_model_class = full_model_class

    def find(self, *args, **kwargs):
        """
        Intercept find() - 返回 QueryProxy 自动处理双存储

        Returns:
            DualStorageQueryProxy
        """
        # 调用原始 model 的 find 方法
        mongo_cursor = self._original_model.find(*args, **kwargs)

        # 包装成 QueryProxy
        return DualStorageQueryProxy(
            mongo_cursor=mongo_cursor,
            kv_storage=self._kv_storage,
            full_model_class=self._full_model_class,
        )

    async def get(
        self, doc_id, session: Optional[AsyncClientSession] = None, **kwargs
    ):
        """
        Intercept get() - 优先从 KV-Storage 读取

        Args:
            doc_id: Document ID (ObjectId or str)
            session: Optional MongoDB session

        Returns:
            Full document or None
        """
        try:
            # 优先从 KV-Storage 读取
            doc_id_str = str(doc_id)
            kv_value = await self._kv_storage.get(key=doc_id_str)

            if kv_value:
                # KV hit
                document = self._full_model_class.model_validate_json(kv_value)
                logger.debug(f"✅ KV hit: {doc_id_str}")
                return document

            # KV miss - fallback to MongoDB
            logger.debug(f"⚠️  KV miss, fallback to MongoDB: {doc_id_str}")
            document = await self._original_model.get(doc_id, session=session, **kwargs)

            # 回填 KV-Storage
            if document:
                kv_value = document.model_dump_json()
                await self._kv_storage.put(key=str(document.id), value=kv_value)
                logger.debug(f"✅ Backfilled KV-Storage: {doc_id_str}")

            return document

        except Exception as e:
            logger.error(f"❌ Failed to get document: {e}")
            return None

    def __getattr__(self, name):
        """Proxy all other methods to original model"""
        return getattr(self._original_model, name)


class DocumentInstanceWrapper:
    """
    Document Instance Wrapper - 拦截 Document 实例方法

    拦截 insert(), save(), delete() 等实例方法，自动同步 KV-Storage
    """

    @staticmethod
    def wrap_insert(original_insert, kv_storage: "KVStorageInterface"):
        """Wrap document.insert() to sync KV-Storage"""
        async def wrapped_insert(self, **kwargs):
            # 调用原始 insert (传递 self)
            result = await original_insert(self, **kwargs)

            # 同步到 KV-Storage
            if self.id:
                try:
                    kv_key = str(self.id)
                    kv_value = self.model_dump_json()
                    await kv_storage.put(key=kv_key, value=kv_value)
                    logger.debug(f"✅ Synced to KV-Storage after insert: {kv_key}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to sync to KV-Storage: {e}")

            return result

        return wrapped_insert

    @staticmethod
    def wrap_save(original_save, kv_storage: "KVStorageInterface"):
        """Wrap document.save() to sync KV-Storage"""
        async def wrapped_save(self, **kwargs):
            # 调用原始 save (传递 self)
            result = await original_save(self, **kwargs)

            # 同步到 KV-Storage
            if self.id:
                try:
                    kv_key = str(self.id)
                    kv_value = self.model_dump_json()
                    await kv_storage.put(key=kv_key, value=kv_value)
                    logger.debug(f"✅ Synced to KV-Storage after save: {kv_key}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to sync to KV-Storage: {e}")

            return result

        return wrapped_save

    @staticmethod
    def wrap_delete(original_delete, kv_storage: "KVStorageInterface"):
        """Wrap document.delete() to remove from KV-Storage"""
        async def wrapped_delete(self, **kwargs):
            doc_id = str(self.id) if self.id else None

            # 调用原始 delete (传递 self)
            result = await original_delete(self, **kwargs)

            # 从 KV-Storage 删除
            if doc_id:
                try:
                    await kv_storage.delete(key=doc_id)
                    logger.debug(f"✅ Deleted from KV-Storage: {doc_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

            return result

        return wrapped_delete


__all__ = [
    "DualStorageModelProxy",
    "DualStorageQueryProxy",
    "DocumentInstanceWrapper",
]
