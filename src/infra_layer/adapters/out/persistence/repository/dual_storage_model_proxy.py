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

    async def count(self, *args, **kwargs):
        """Proxy count method to original cursor"""
        return await self._mongo_cursor.count(*args, **kwargs)

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

    async def find_one(self, *args, **kwargs):
        """
        Intercept find_one() - 优先从 KV 读取（如果有 _id 过滤）

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Document or None
        """
        # 调用原始 find_one
        document = await self._original_model.find_one(*args, **kwargs)

        # 如果找到文档，回填 KV-Storage
        if document and document.id:
            try:
                kv_key = str(document.id)
                kv_value = document.model_dump_json()
                await self._kv_storage.put(key=kv_key, value=kv_value)
                logger.debug(f"✅ Backfilled KV-Storage from find_one: {kv_key}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to backfill KV-Storage: {e}")

        return document

    async def delete_many(self, *args, **kwargs):
        """
        Intercept delete_many() - 批量删除并同步 KV-Storage

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Delete result
        """
        try:
            # 1. 先查询要删除的文档 IDs
            filter_query = args[0] if args else {}
            docs = await self._original_model.find(filter_query).to_list()
            doc_ids = [str(doc.id) for doc in docs]

            # 2. 执行删除
            result = await self._original_model.delete_many(*args, **kwargs)

            # 3. 批量删除 KV-Storage
            if doc_ids:
                try:
                    await self._kv_storage.batch_delete(keys=doc_ids)
                    logger.debug(f"✅ Deleted {len(doc_ids)} documents from KV-Storage")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to delete_many with dual storage: {e}")
            raise

    def hard_find_one(self, *args, **kwargs):
        """
        Intercept hard_find_one() - 查询包括已删除的文档，并回填 KV

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            FindOne query object
        """
        # hard_find_one returns a query object, we need to wrap it
        # But since it's a class method returning a query object, we'll just pass through
        # and handle backfill in the wrapper if needed
        return self._original_model.hard_find_one(*args, **kwargs)

    async def hard_delete_many(self, *args, **kwargs):
        """
        Intercept hard_delete_many() - 物理删除并同步 KV-Storage

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Delete result
        """
        try:
            # 1. 先查询要删除的文档 IDs (使用 hard_find_many 查询所有包括已删除的)
            filter_query = args[0] if args else {}
            docs = await self._original_model.hard_find_many(filter_query).to_list()
            doc_ids = [str(doc.id) for doc in docs]

            # 2. 执行物理删除
            result = await self._original_model.hard_delete_many(*args, **kwargs)

            # 3. 批量删除 KV-Storage
            if doc_ids:
                try:
                    await self._kv_storage.batch_delete(keys=doc_ids)
                    logger.debug(f"✅ Hard deleted {len(doc_ids)} documents from KV-Storage")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to hard_delete_many with dual storage: {e}")
            raise

    async def restore_many(self, *args, **kwargs):
        """
        Intercept restore_many() - 恢复已删除文档并同步 KV-Storage

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Update result
        """
        try:
            # 1. 执行恢复操作
            result = await self._original_model.restore_many(*args, **kwargs)

            # 2. 查询被恢复的文档并同步到 KV-Storage
            filter_query = args[0] if args else {}
            # 恢复后的文档 deleted_at = None，查询它们
            restored_docs = await self._original_model.find(filter_query).to_list()

            # 3. 批量同步到 KV-Storage
            if restored_docs:
                try:
                    for doc in restored_docs:
                        kv_key = str(doc.id)
                        kv_value = doc.model_dump_json()
                        await self._kv_storage.put(key=kv_key, value=kv_value)
                    logger.debug(f"✅ Restored {len(restored_docs)} documents to KV-Storage")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to restore to KV-Storage: {e}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to restore_many with dual storage: {e}")
            raise

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

    @staticmethod
    def wrap_restore(original_restore, kv_storage: "KVStorageInterface"):
        """Wrap document.restore() to sync back to KV-Storage"""
        async def wrapped_restore(self, **kwargs):
            # 调用原始 restore (传递 self)
            result = await original_restore(self, **kwargs)

            # 恢复后同步回 KV-Storage
            if self.id:
                try:
                    kv_key = str(self.id)
                    kv_value = self.model_dump_json()
                    await kv_storage.put(key=kv_key, value=kv_value)
                    logger.debug(f"✅ Synced to KV-Storage after restore: {kv_key}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to sync to KV-Storage after restore: {e}")

            return result

        return wrapped_restore

    @staticmethod
    def wrap_hard_delete(original_hard_delete, kv_storage: "KVStorageInterface"):
        """Wrap document.hard_delete() to remove from KV-Storage"""
        async def wrapped_hard_delete(self, **kwargs):
            doc_id = str(self.id) if self.id else None

            # 调用原始 hard_delete (传递 self)
            result = await original_hard_delete(self, **kwargs)

            # 从 KV-Storage 删除
            if doc_id:
                try:
                    await kv_storage.delete(key=doc_id)
                    logger.debug(f"✅ Deleted from KV-Storage after hard_delete: {doc_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage after hard_delete: {e}")

            return result

        return wrapped_hard_delete


__all__ = [
    "DualStorageModelProxy",
    "DualStorageQueryProxy",
    "DocumentInstanceWrapper",
]
