"""
Dual Storage Mixin - 自动拦截 CRUD 方法并同步到 KV-Storage

最小化侵入方案：
- Repository 只需添加此 Mixin 到继承列表
- 自动拦截 append, get_by_id, update_by_id, delete_by_id
- 自动同步完整文档到 KV-Storage
- MongoDB 保留所有字段和索引（与主分支一致）

使用示例：
    class EpisodicMemoryRawRepository(
        DualStorageMixin[EpisodicMemory],  # 只需加这一行
        BaseRepository[EpisodicMemory]
    ):
        # 原有代码完全不变
        pass
"""

from typing import TypeVar, Generic, List, Optional, Dict, Any
from pymongo.asynchronous.client_session import AsyncClientSession
from bson import ObjectId

from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
    KVStorageInterface,
)

logger = get_logger(__name__)

TDocument = TypeVar("TDocument")


class DualStorageMixin(Generic[TDocument]):
    """
    Dual Storage Mixin - 自动拦截 CRUD 并同步 KV-Storage

    工作原理：
    1. 拦截 append/update - 写入 MongoDB 后自动同步到 KV-Storage
    2. 拦截 get_by_id - 优先从 KV-Storage 读取（快速）
    3. 拦截 delete - 删除 MongoDB 后自动从 KV-Storage 删除
    4. find_by_filter - 从 MongoDB 查询索引，按需从 KV 加载完整数据

    优势：
    - Repository 代码几乎零改动
    - Document 模型完全不变
    - 自动处理双存储同步
    """

    def __init__(self, *args, **kwargs):
        """Initialize mixin and get KV-Storage instance"""
        super().__init__(*args, **kwargs)
        self._kv_storage: Optional[KVStorageInterface] = None

    def _get_kv_storage(self) -> KVStorageInterface:
        """Lazy load KV-Storage instance from DI container"""
        if self._kv_storage is None:
            from core.di import get_bean_by_type

            self._kv_storage = get_bean_by_type(KVStorageInterface)
        return self._kv_storage

    async def append(
        self, document: TDocument, session: Optional[AsyncClientSession] = None
    ) -> Optional[TDocument]:
        """
        Append document - 写入 MongoDB 后自动同步到 KV-Storage

        便利方法：内部调用 document.insert() + KV sync
        Repository 的业务方法可以调用此方法来自动启用双存储

        Args:
            document: Document to append
            session: Optional MongoDB session

        Returns:
            Appended document with ID set
        """
        try:
            # 调用 Beanie Document 的 insert 方法 (写入 MongoDB)
            await document.insert(session=session)

            # 同步完整文档到 KV-Storage
            if document.id:
                kv_storage = self._get_kv_storage()
                kv_key = str(document.id)
                kv_value = document.model_dump_json()
                await kv_storage.put(key=kv_key, value=kv_value)
                logger.debug(f"✅ Synced to KV-Storage: {kv_key}")

            return document

        except Exception as e:
            logger.error(f"❌ Failed to append with dual storage: {e}")
            raise

    async def get_by_id(
        self, doc_id: str, session: Optional[AsyncClientSession] = None
    ) -> Optional[TDocument]:
        """
        覆盖 get_by_id - 优先从 KV-Storage 读取（快速路径）

        Args:
            doc_id: Document ID
            session: Optional MongoDB session

        Returns:
            Document or None
        """
        try:
            # 优先从 KV-Storage 读取（快速）
            kv_storage = self._get_kv_storage()
            kv_value = await kv_storage.get(key=doc_id)

            if kv_value:
                # KV hit - 直接反序列化返回
                document = self.model.model_validate_json(kv_value)
                logger.debug(f"✅ KV hit: {doc_id}")
                return document

            # KV miss - fallback 到 MongoDB (调用 BaseRepository 的方法)
            logger.debug(f"⚠️  KV miss, fallback to MongoDB: {doc_id}")
            # 使用 self.model 直接查询 MongoDB
            from bson import ObjectId
            try:
                object_id = ObjectId(doc_id) if not isinstance(doc_id, ObjectId) else doc_id
            except:
                return None

            result = await self.model.get(object_id, session=session)

            # 回填 KV-Storage
            if result:
                kv_key = str(result.id)
                kv_value = result.model_dump_json()
                await kv_storage.put(key=kv_key, value=kv_value)
                logger.debug(f"✅ Backfilled KV-Storage: {kv_key}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to get_by_id with dual storage: {e}")
            return None

    async def update_by_id(
        self,
        doc_id: str,
        update_data: Dict[str, Any],
        session: Optional[AsyncClientSession] = None,
    ) -> Optional[TDocument]:
        """
        Update by ID - 更新 MongoDB 后自动同步到 KV-Storage

        Args:
            doc_id: Document ID
            update_data: Fields to update
            session: Optional MongoDB session

        Returns:
            Updated document or None
        """
        try:
            # 从 MongoDB 获取文档
            from bson import ObjectId
            try:
                object_id = ObjectId(doc_id) if not isinstance(doc_id, ObjectId) else doc_id
            except:
                return None

            document = await self.model.get(object_id, session=session)
            if not document:
                return None

            # 更新字段
            for key, value in update_data.items():
                if hasattr(document, key):
                    setattr(document, key, value)

            # 保存到 MongoDB
            await document.save(session=session)

            # 同步到 KV-Storage
            kv_storage = self._get_kv_storage()
            kv_key = str(document.id)
            kv_value = document.model_dump_json()
            await kv_storage.put(key=kv_key, value=kv_value)
            logger.debug(f"✅ Updated KV-Storage: {kv_key}")

            return document

        except Exception as e:
            logger.error(f"❌ Failed to update_by_id with dual storage: {e}")
            return None

    async def delete_by_id(
        self, doc_id: str, session: Optional[AsyncClientSession] = None
    ) -> bool:
        """
        Delete by ID - 删除 MongoDB 后自动从 KV-Storage 删除

        Args:
            doc_id: Document ID
            session: Optional MongoDB session

        Returns:
            True if deleted, False otherwise
        """
        try:
            # 从 MongoDB 获取并删除
            from bson import ObjectId
            try:
                object_id = ObjectId(doc_id) if not isinstance(doc_id, ObjectId) else doc_id
            except:
                return False

            document = await self.model.get(object_id, session=session)
            if not document:
                return False

            # 删除 MongoDB
            await document.delete(session=session)

            # 从 KV-Storage 删除
            kv_storage = self._get_kv_storage()
            await kv_storage.delete(key=doc_id)
            logger.debug(f"✅ Deleted from KV-Storage: {doc_id}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete_by_id with dual storage: {e}")
            return False

    async def find_by_filter(
        self,
        query_filter: Optional[Dict[str, Any]] = None,
        skip: int = 0,
        limit: Optional[int] = None,
        sort_field: str = "created_at",
        sort_desc: bool = False,
        session: Optional[AsyncClientSession] = None,
    ) -> List[TDocument]:
        """
        Query by filter - 从 MongoDB 查询

        Args:
            query_filter: MongoDB query filter
            skip: Results to skip
            limit: Max results
            sort_field: Field to sort by
            sort_desc: Sort descending if True
            session: Optional MongoDB session

        Returns:
            List of documents
        """
        try:
            # 从 MongoDB 查询（利用索引）
            query = self.model.find(query_filter or {}, session=session)

            # 排序
            if sort_desc:
                query = query.sort(f"-{sort_field}")
            else:
                query = query.sort(sort_field)

            # 分页
            if skip:
                query = query.skip(skip)
            if limit:
                query = query.limit(limit)

            # 执行查询
            results = await query.to_list()

            # MongoDB 已经有完整数据，直接返回
            # （因为主分支的 EpisodicMemory 在 MongoDB 中存储了所有字段）
            return results

        except Exception as e:
            logger.error(f"❌ Failed to find_by_filter with dual storage: {e}")
            return []


__all__ = ["DualStorageMixin"]
